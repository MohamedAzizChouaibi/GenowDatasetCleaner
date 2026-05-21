#!/usr/bin/env python3
"""GenowDatasetCleaner — YOLOv11/v12 image dataset cleaner with VLM-based dedup.

Supported dataset formats: YOLOv11 / YOLOv12 (Ultralytics).
A dataset is accepted if it contains either a `data.yaml` with `train:` + `names:`/`nc:`,
or parallel `images/` and `labels/` directories.

Usage:
    python genowCleaner.py                 # launch GUI
    python genowCleaner.py --cli DATASET   # headless analysis, writes JSON report
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import re
import shutil
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image

from embedders import AVAILABLE_EMBEDDERS, DEFAULT_EMBEDDER_KEY, get_spec, load_embedder

try:
    from send2trash import send2trash as _send2trash
    HAS_SEND2TRASH = True
except ImportError:
    HAS_SEND2TRASH = False

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

logger = logging.getLogger("genow_cleaner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAME_FILTER_RE = re.compile(r"(?:^|[_\-.])(det|seg)(?:[_\-.]|$)", re.IGNORECASE)
SETTINGS_FILE = Path.home() / ".genow_cleaner_settings.json"
APP_DIR = Path.home() / ".genow_cleaner"
THUMB_SIZE = (180, 180)
LOGO_PATH = Path(__file__).parent / "genow_logo.jpeg"

COLOR_NAVY = "#1a1f3a"
COLOR_NAVY_DEEP = "#0f1428"
COLOR_MAGENTA = "#e8336d"
COLOR_MAGENTA_HOVER = "#c92458"
COLOR_BG = "#ffffff"
COLOR_SURFACE = "#ffffff"
COLOR_BORDER = "#e4e7ef"
COLOR_TEXT = "#1a1f3a"
COLOR_TEXT_MUTED = "#6b7280"
COLOR_DANGER = "#dc2626"
COLOR_SUCCESS = "#16a34a"


@dataclass
class Config:
    blur_threshold: float = 60.0
    dark_threshold: float = 10.0
    bright_threshold: float = 230.0
    low_info_threshold: float = 12.0
    duplicate_threshold: float = 0.96
    batch_size: int = 32
    enable_name_filter: bool = True
    use_trash: bool = True
    embedder: str = DEFAULT_EMBEDDER_KEY
    workers: int = 4
    phash_prepass: bool = True

    def save(self) -> None:
        try:
            SETTINGS_FILE.write_text(json.dumps(asdict(self), indent=2))
        except OSError as e:
            logger.warning("Could not save settings: %s", e)

    @classmethod
    def load(cls) -> "Config":
        if SETTINGS_FILE.exists():
            try:
                data = json.loads(SETTINGS_FILE.read_text())
                fields = cls.__dataclass_fields__
                return cls(**{k: v for k, v in data.items() if k in fields})
            except (OSError, json.JSONDecodeError, TypeError) as e:
                logger.warning("Settings load failed (%s), using defaults", e)
        return cls()


# ---------- Analysis primitives ----------

def _read_image_bgr(path: str) -> Optional[np.ndarray]:
    """Read an image to BGR, with PIL fallback for non-ASCII paths and exotic formats."""
    img = cv2.imread(path)
    if img is not None:
        return img
    try:
        with Image.open(path) as raw:
            arr = np.array(raw.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def analyze_image_quality(path: str) -> Optional[dict]:
    img = _read_image_bgr(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    return {
        "blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
        "std": float(gray.std()),
        "pixels": int(h * w),
    }


def quality_score(m: dict) -> float:
    """Composite quality score: higher = better candidate to keep."""
    if not m:
        return float("-inf")
    bright_penalty = abs(m.get("brightness", 128.0) - 128.0) / 128.0
    return (
        float(np.log1p(m.get("blur", 0.0)))
        + float(np.log1p(m.get("std", 0.0)))
        + 0.5 * float(np.log1p(m.get("pixels", 0)))
        - bright_penalty
    )


def collect_images(dataset_path: Path) -> list[Path]:
    """Recursively collect image files without following symlinks.

    Skips the YOLO `labels/` subtrees so label-adjacent .txt files don't get
    scanned as images, and skips quarantine/report dirs.
    """
    out: list[Path] = []
    for root, dirs, files in os.walk(dataset_path, followlinks=False):
        dirs[:] = [d for d in dirs
                   if not d.startswith("_quarantine_")
                   and not d.startswith(".genow_cleaner")
                   and d.lower() != "labels"]
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in IMG_EXTENSIONS:
                out.append(p)
    return out


# ---------- YOLOv11 / v12 dataset support ----------

def validate_yolo_dataset(path: Path) -> tuple[bool, str]:
    """Return (ok, reason). Accepts either a data.yaml with YOLO keys,
    or a parallel images/+labels/ layout (Ultralytics convention)."""
    if not path.is_dir():
        return False, f"Not a directory: {path}"
    yamls = list(path.glob("*.yaml")) + list(path.glob("*.yml"))
    for y in yamls:
        try:
            text = y.read_text(errors="ignore")
        except OSError:
            continue
        has_train = "train:" in text or "train :" in text
        has_classes = "names:" in text or "nc:" in text
        if has_train and has_classes:
            return True, f"data file: {y.name}"
    if (path / "images").is_dir() and (path / "labels").is_dir():
        return True, "images/ + labels/ layout"
    return False, ("No YOLOv11/v12 dataset detected — expected a data.yaml with "
                   "`train:` and `names:`/`nc:`, or parallel images/ and labels/ "
                   "directories.")


def yolo_label_path(image_path: Path) -> Path:
    """Resolve the YOLO label file for an image.

    `images/<split>/foo.jpg` -> `labels/<split>/foo.txt`. If no `images`
    segment is found, falls back to a sibling .txt.
    """
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def cluster_by_similarity(embeddings: np.ndarray, threshold: float,
                          cancel_check: Callable[[], bool] = lambda: False,
                          progress: Optional[Callable[[int, int], None]] = None
                          ) -> list[list[int]]:
    """Cluster row-normalized embeddings by cosine similarity > threshold.

    Memory is O(chunk * N), not O(N^2). Time is O(N^2 / chunk * chunk) = O(N^2).
    """
    n = embeddings.shape[0]
    if n < 2:
        return []
    uf = UnionFind(n)
    chunk = 512
    for s in range(0, n, chunk):
        if cancel_check():
            return []
        e = min(n, s + chunk)
        sim = embeddings[s:e] @ embeddings.T
        for li, i in enumerate(range(s, e)):
            row = sim[li, i + 1:]
            hits = np.where(row > threshold)[0] + (i + 1)
            for j in hits:
                uf.union(i, int(j))
        if progress:
            progress(e, n)
    groups: dict[int, list[int]] = {}
    for idx in range(n):
        groups.setdefault(uf.find(idx), []).append(idx)
    return [m for m in groups.values() if len(m) >= 2]


def perceptual_hash_groups(paths: list[Path],
                            cancel_check: Callable[[], bool] = lambda: False,
                            progress: Optional[Callable[[int, int], None]] = None,
                            workers: int = 4
                            ) -> dict[str, list[Path]]:
    """Group paths by identical perceptual hash. Cheap pre-pass before CLIP."""
    if not HAS_IMAGEHASH:
        return {}
    groups: dict[str, list[Path]] = {}
    lock = threading.Lock()
    n = len(paths)
    done = [0]

    def task(p: Path):
        if cancel_check():
            return None
        try:
            with Image.open(p) as img:
                h = str(imagehash.phash(img.convert("RGB")))
        except (OSError, ValueError):
            h = None
        with lock:
            done[0] += 1
            if progress and (done[0] % 50 == 0 or done[0] == n):
                progress(done[0], n)
        return (p, h)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(task, paths):
            if res is None:
                continue
            p, h = res
            if h is None:
                continue
            groups.setdefault(h, []).append(p)
    return groups


# ---------- Background job ----------

@dataclass
class Event:
    kind: str
    value: float = 0.0
    message: str = ""
    payload: object = None


class AnalysisJob(threading.Thread):
    def __init__(self, dataset_path: Path, config: Config, event_q: queue.Queue,
                 embedder):
        super().__init__(daemon=True)
        self.dataset_path = dataset_path
        self.config = config
        self.event_q = event_q
        self.embedder = embedder
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def _post(self, kind: str, **kwargs) -> None:
        self.event_q.put(Event(kind=kind, **kwargs))

    def run(self) -> None:
        try:
            results = self._analyze()
            if self.cancelled():
                self._post("cancelled")
            else:
                self._post("done", payload=results)
        except Exception as e:
            logger.exception("Analysis failed")
            self._post("error", message=str(e))

    def _analyze(self) -> dict:
        ok, reason = validate_yolo_dataset(self.dataset_path)
        if not ok:
            raise RuntimeError(reason)
        self._post("log", message=f"YOLOv11/v12 dataset OK ({reason}).")
        self._post("log", message="Collecting image files…")
        all_imgs = collect_images(self.dataset_path)
        self._post("log", message=f"Found {len(all_imgs)} image file(s).")

        name_filtered: list[Path] = []
        clean: list[Path] = []
        if self.config.enable_name_filter:
            for p in all_imgs:
                (name_filtered if NAME_FILTER_RE.search(p.stem) else clean).append(p)
        else:
            clean = all_imgs

        quality, unreadable = self._analyze_quality(clean)
        if self.cancelled():
            return {}

        blurry: list[tuple] = []
        dark: list[tuple] = []
        bright: list[tuple] = []
        low_info: list[tuple] = []
        for p, m in quality.items():
            if m["blur"] < self.config.blur_threshold:
                blurry.append((p, m["blur"]))
            if m["brightness"] < self.config.dark_threshold:
                dark.append((p, m["brightness"]))
            elif m["brightness"] > self.config.bright_threshold:
                bright.append((p, m["brightness"]))
            if m["std"] < self.config.low_info_threshold:
                low_info.append((p, m["std"]))

        clusters: list[list[Path]] = []
        if clean and not self.cancelled():
            # Perceptual-hash pre-pass: groups exact/near-exact dupes cheaply
            phash_dupes: list[list[Path]] = []
            paths_for_clip = clean
            if self.config.phash_prepass and HAS_IMAGEHASH:
                self._post("log", message="Perceptual-hash pre-pass…")
                groups = perceptual_hash_groups(
                    clean,
                    cancel_check=self.cancelled,
                    progress=lambda d, n: self._post(
                        "progress", value=40 + d / max(n, 1) * 10,
                        message=f"pHash {d}/{n}"),
                    workers=self.config.workers,
                )
                exact_dupes: set[Path] = set()
                for hash_val, members in groups.items():
                    if len(members) >= 2:
                        members.sort(key=lambda p: quality_score(quality.get(p, {})),
                                     reverse=True)
                        phash_dupes.append(members)
                        exact_dupes.update(members[1:])
                paths_for_clip = [p for p in clean if p not in exact_dupes]
                if phash_dupes:
                    self._post("log",
                               message=f"pHash found {len(phash_dupes)} exact-dup group(s); "
                                       f"{len(paths_for_clip)} unique image(s) for CLIP.")

            embeddings, valid = self._embed(paths_for_clip)
            if self.cancelled():
                return {}
            clip_clusters_idx = cluster_by_similarity(
                embeddings, self.config.duplicate_threshold,
                cancel_check=self.cancelled,
                progress=lambda done, total: self._post(
                    "progress", value=90 + done / max(total, 1) * 10,
                    message=f"Dedup {done}/{total}"),
            )
            clip_clusters: list[list[Path]] = []
            for members in clip_clusters_idx:
                ps = [valid[m] for m in members]
                ps.sort(key=lambda p: quality_score(quality.get(p, {})), reverse=True)
                clip_clusters.append(ps)
            clusters = phash_dupes + clip_clusters
            clusters.sort(key=len, reverse=True)

        scores = {str(p): quality_score(m) for p, m in quality.items()}
        return {
            "blurry": blurry,
            "dark": dark,
            "bright": bright,
            "low_info": low_info,
            "name_filtered": name_filtered,
            "unreadable": unreadable,
            "duplicate_clusters": clusters,
            "quality_scores": scores,
            "total_scanned": len(all_imgs),
        }

    def _analyze_quality(self, paths: list[Path]) -> tuple[dict[Path, dict], list[Path]]:
        quality: dict[Path, dict] = {}
        unreadable: list[Path] = []
        n = len(paths)
        if n == 0:
            return quality, unreadable
        lock = threading.Lock()
        done = [0]

        def task(p: Path):
            if self.cancelled():
                return p, None
            m = analyze_image_quality(str(p))
            with lock:
                done[0] += 1
                if done[0] % 20 == 0 or done[0] == n:
                    self._post("progress",
                               value=done[0] / n * 40,
                               message=f"Quality {done[0]}/{n}")
            return p, m

        workers = max(1, int(self.config.workers))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for p, m in ex.map(task, paths):
                if self.cancelled():
                    break
                if m is None:
                    unreadable.append(p)
                else:
                    quality[p] = m
        return quality, unreadable

    def _embed(self, paths: list[Path]):
        n = len(paths)
        feats_chunks: list[np.ndarray] = []
        valid_paths: list[Path] = []
        bs = max(1, int(self.config.batch_size))
        model_label = self.embedder.spec.label
        for start in range(0, n, bs):
            if self.cancelled():
                break
            batch = paths[start:start + bs]
            pil_imgs: list = []
            ok_paths: list[Path] = []
            for p in batch:
                try:
                    with Image.open(p) as raw:
                        pil_imgs.append(raw.convert("RGB"))
                    ok_paths.append(p)
                except (OSError, ValueError) as e:
                    self._post("log", message=f"Skip embedding for {p.name}: {e}")
            if pil_imgs:
                feats = self.embedder.embed(pil_imgs)
                feats_chunks.append(feats)
                valid_paths.extend(ok_paths)
            done = min(start + bs, n)
            self._post("progress", value=50 + done / max(n, 1) * 40,
                       message=f"{model_label} embeddings {done}/{n}")
        if not feats_chunks:
            return np.zeros((0, 0), dtype=np.float32), []
        return np.vstack(feats_chunks), valid_paths


# ---------- File operations ----------

def _safe_move(src: Path, dst_dir: Path) -> Optional[Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / src.name
    stem, suf = target.stem, target.suffix
    i = 1
    while target.exists():
        target = dst_dir / f"{stem}_{i}{suf}"
        i += 1
    try:
        shutil.move(str(src), target)
        return target
    except OSError as e:
        logger.error("Failed to move %s: %s", src, e)
        return None


def remove_yolo_pair(image_path: Path, use_trash: bool, quarantine_dir: Path
                     ) -> tuple[int, list[tuple[Path, Path]]]:
    """Remove a YOLO image and its companion label .txt.

    Returns (count_removed, undo_moves). undo_moves is populated only when files
    were moved to quarantine (system-trash deletes are not tracked for undo).
    """
    moves: list[tuple[Path, Path]] = []
    count = 0
    targets = [image_path, yolo_label_path(image_path)]
    for t in targets:
        if not t.exists():
            continue
        if use_trash and HAS_SEND2TRASH:
            try:
                _send2trash(str(t))
                count += 1
            except OSError as e:
                logger.error("Trash failed for %s: %s", t, e)
        else:
            moved_to = _safe_move(t, quarantine_dir)
            if moved_to is not None:
                count += 1
                moves.append((t, moved_to))
    return count, moves


# ---------- JSON report ----------

def write_report(report_path: Path, dataset: Path, results: dict, config: Config) -> None:
    def _items(key):
        out = []
        for it in results.get(key, []):
            if isinstance(it, tuple):
                p, score = it
                out.append({"path": str(p), "score": float(score)})
            else:
                out.append({"path": str(it)})
        return out

    clusters_out = []
    for c in results.get("duplicate_clusters", []):
        scores = results.get("quality_scores", {})
        clusters_out.append([
            {"path": str(p), "quality_score": scores.get(str(p))} for p in c
        ])

    report = {
        "schema": 2,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "dataset_format": "YOLOv11/v12",
        "config": asdict(config),
        "total_scanned": results.get("total_scanned", 0),
        "categories": {
            "blurry": _items("blurry"),
            "dark": _items("dark"),
            "bright": _items("bright"),
            "low_info": _items("low_info"),
            "name_filtered": _items("name_filtered"),
            "unreadable": _items("unreadable"),
        },
        "duplicate_clusters": clusters_out,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))


# ---------- GUI (imported lazily so --cli works without a display) ----------

def _run_gui() -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- LRU thumbnail cache -----
    class ThumbCache:
        def __init__(self, max_size: int = 500) -> None:
            self._cache: "OrderedDict[Path, ImageTk.PhotoImage]" = OrderedDict()
            self._max = max_size

        def get(self, path: Path):
            if path in self._cache:
                self._cache.move_to_end(path)
                return self._cache[path]
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    img.thumbnail(THUMB_SIZE)
                    photo = ImageTk.PhotoImage(img)
            except (OSError, ValueError):
                return None
            self._cache[path] = photo
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)
            return photo

    # ----- Scrollable container -----
    class ScrollableFrame(ttk.Frame):
        def __init__(self, master, **kw):
            super().__init__(master, **kw)
            self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg=COLOR_BG)
            self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=self.vsb.set)
            self.vsb.pack(side="right", fill="y")
            self.canvas.pack(side="left", fill="both", expand=True)
            self.inner = ttk.Frame(self.canvas)
            self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
            self.inner.bind("<Configure>",
                            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
            self.canvas.bind("<Configure>",
                             lambda e: self.canvas.itemconfigure(self.inner_id, width=e.width))
            self.canvas.bind("<Enter>", self._bind_wheel)
            self.canvas.bind("<Leave>", self._unbind_wheel)

        def _on_wheel(self, evt):
            if getattr(evt, "num", None) == 4:
                delta = -1
            elif getattr(evt, "num", None) == 5:
                delta = 1
            else:
                delta = -1 if evt.delta > 0 else 1
            self.canvas.yview_scroll(delta, "units")

        def _bind_wheel(self, _e):
            self.canvas.bind_all("<MouseWheel>", self._on_wheel)
            self.canvas.bind_all("<Button-4>", self._on_wheel)
            self.canvas.bind_all("<Button-5>", self._on_wheel)

        def _unbind_wheel(self, _e):
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")

    # ----- Review windows -----
    class CategoryReview(tk.Toplevel):
        def __init__(self, master, title, items, score_label, on_delete, thumb_cache):
            super().__init__(master)
            self.title(f"Review — {title}")
            self.geometry("1000x720")
            self.configure(bg=COLOR_BG)
            self.items = items
            self.score_label = score_label
            self.on_delete = on_delete
            self.thumbs = thumb_cache
            self.vars = []
            self.paths = []
            self._build(title)

        def _build(self, title):
            hdr = ttk.Frame(self, padding=12)
            hdr.pack(fill="x")
            ttk.Label(hdr, text=title, style="Header.TLabel").pack(side="left")
            ttk.Label(hdr, text=f"   {len(self.items)} item(s)",
                      style="Sub.TLabel").pack(side="left")

            toolbar = ttk.Frame(self, padding=(12, 0))
            toolbar.pack(fill="x")
            ttk.Button(toolbar, text="Select All",
                       command=lambda: self._set_all(True)).pack(side="left", padx=2)
            ttk.Button(toolbar, text="Select None",
                       command=lambda: self._set_all(False)).pack(side="left", padx=2)
            self.summary_lbl = ttk.Label(toolbar, text="", style="Sub.TLabel")
            self.summary_lbl.pack(side="left", padx=12)

            body = ScrollableFrame(self)
            body.pack(fill="both", expand=True, padx=12, pady=10)
            grid = body.inner
            cols = 5
            for i, item in enumerate(self.items):
                if isinstance(item, tuple):
                    path, score = item
                    caption = f"{self.score_label}: {score:.2f}"
                else:
                    path, caption = item, ""
                self.paths.append(path)
                var = tk.BooleanVar(value=True)
                var.trace_add("write", lambda *_: self._update_summary())
                self.vars.append(var)

                card = ttk.Frame(grid, style="Card.TFrame", padding=6)
                card.grid(row=i // cols, column=i % cols, padx=6, pady=6, sticky="nsew")
                grid.columnconfigure(i % cols, weight=1)

                photo = self.thumbs.get(path)
                if photo:
                    lbl = ttk.Label(card, image=photo, style="Card.TLabel")
                    lbl.image = photo
                else:
                    lbl = ttk.Label(card, text="(unreadable)", style="Card.TLabel")
                lbl.pack()
                ttk.Label(card, text=path.name, style="Card.TLabel",
                          wraplength=170).pack(pady=(4, 0))
                if caption:
                    ttk.Label(card, text=caption, style="Card.TLabel",
                              foreground=COLOR_TEXT_MUTED).pack()
                ttk.Checkbutton(card, text="Delete", variable=var).pack(pady=(4, 0))

            footer = ttk.Frame(self, padding=12)
            footer.pack(fill="x")
            ttk.Button(footer, text="Close", command=self.destroy).pack(side="right", padx=4)
            ttk.Button(footer, text="Delete Selected", style="Danger.TButton",
                       command=self._confirm).pack(side="right", padx=4)
            self._update_summary()

        def _set_all(self, val):
            for v in self.vars:
                v.set(val)

        def _update_summary(self):
            n = sum(1 for v in self.vars if v.get())
            self.summary_lbl.config(text=f"{n}/{len(self.vars)} selected for deletion")

        def _confirm(self):
            to_delete = [self.paths[i] for i, v in enumerate(self.vars) if v.get()]
            if not to_delete:
                messagebox.showinfo("Nothing selected", "No items selected.")
                return
            if not messagebox.askyesno(
                    "Confirm deletion",
                    f"Remove {len(to_delete)} item(s)?\n"
                    f"Companion label files (xml/txt/json) will be removed as well."):
                return
            deleted = self.on_delete(to_delete)
            messagebox.showinfo("Done", f"{deleted} file(s) removed.")
            self.destroy()

    class DuplicateReview(tk.Toplevel):
        def __init__(self, master, clusters, quality_scores, on_delete,
                     on_auto_clean, thumb_cache):
            super().__init__(master)
            self.title("Review — Duplicate clusters")
            self.geometry("1120x800")
            self.configure(bg=COLOR_BG)
            self.clusters = clusters
            self.scores = quality_scores
            self.on_delete = on_delete
            self.on_auto_clean = on_auto_clean
            self.thumbs = thumb_cache
            self.keep_vars = []
            self.skip_vars = []
            self._build()

        def _build(self):
            hdr = ttk.Frame(self, padding=12)
            hdr.pack(fill="x")
            ttk.Label(hdr, text="Duplicate clusters", style="Header.TLabel").pack(side="left")
            total_dupes = sum(len(c) - 1 for c in self.clusters)
            ttk.Label(hdr, text=f"   {len(self.clusters)} cluster(s) · {total_dupes} duplicate(s)",
                      style="Sub.TLabel").pack(side="left")

            body = ScrollableFrame(self)
            body.pack(fill="both", expand=True, padx=12, pady=10)
            container = body.inner

            for ci, members in enumerate(self.clusters):
                keep_var = tk.IntVar(value=0)
                skip_var = tk.BooleanVar(value=False)
                self.keep_vars.append(keep_var)
                self.skip_vars.append(skip_var)

                box = ttk.Frame(container, style="Card.TFrame", padding=10)
                box.pack(fill="x", padx=4, pady=6)

                head = ttk.Frame(box, style="Card.TFrame")
                head.pack(fill="x")
                ttk.Label(head, text=f"Cluster {ci + 1} — {len(members)} images",
                          style="CardTitle.TLabel").pack(side="left")
                ttk.Checkbutton(head, text="Skip this cluster (keep all)",
                                variable=skip_var).pack(side="right")

                row = ttk.Frame(box, style="Card.TFrame")
                row.pack(fill="x", pady=(8, 0))
                for mi, path in enumerate(members):
                    card = ttk.Frame(row, style="Card.TFrame", padding=4)
                    card.pack(side="left", padx=4)
                    photo = self.thumbs.get(path)
                    if photo:
                        lbl = ttk.Label(card, image=photo, style="Card.TLabel")
                        lbl.image = photo
                    else:
                        lbl = ttk.Label(card, text="(unreadable)", style="Card.TLabel")
                    lbl.pack()
                    ttk.Label(card, text=path.name, style="Card.TLabel",
                              wraplength=170).pack()
                    score = self.scores.get(str(path))
                    score_txt = f"q={score:.2f}" if score is not None else ""
                    if score_txt:
                        ttk.Label(card, text=score_txt, style="Card.TLabel",
                                  foreground=COLOR_TEXT_MUTED).pack()
                    ttk.Radiobutton(card, text="Keep this", variable=keep_var,
                                    value=mi).pack()

            footer = ttk.Frame(self, padding=12)
            footer.pack(fill="x")
            ttk.Button(footer, text="Close", command=self.destroy).pack(side="right", padx=4)
            ttk.Button(footer, text="Delete Non-Kept", style="Danger.TButton",
                       command=self._confirm).pack(side="right", padx=4)
            ttk.Button(footer, text="Auto-clean (keep best in each)",
                       style="Accent.TButton",
                       command=self._auto_clean).pack(side="left", padx=4)

        def _auto_clean(self):
            self.destroy()
            self.on_auto_clean()

        def _confirm(self):
            to_delete = []
            for ci, members in enumerate(self.clusters):
                if self.skip_vars[ci].get():
                    continue
                keep = self.keep_vars[ci].get()
                for mi, p in enumerate(members):
                    if mi != keep:
                        to_delete.append(p)
            if not to_delete:
                messagebox.showinfo("Nothing to delete", "All clusters skipped.")
                return
            if not messagebox.askyesno("Confirm deletion",
                                       f"Remove {len(to_delete)} duplicate(s)?\n"
                                       f"Companion label files will be removed as well."):
                return
            deleted = self.on_delete(to_delete)
            messagebox.showinfo("Done", f"{deleted} file(s) removed.")
            self.destroy()

    CATEGORY_META = [
        ("duplicates", "Duplicates", "DUP"),
        ("blurry", "Blurry", "BLR"),
        ("dark", "Too dark", "DRK"),
        ("bright", "Too bright", "BRT"),
        ("low_info", "Low information", "LOW"),
        ("name_filtered", "Name-filtered (det/seg)", "TAG"),
        ("unreadable", "Unreadable", "ERR"),
    ]
    SCORE_LABELS = {
        "blurry": "Blur",
        "dark": "Brightness",
        "bright": "Brightness",
        "low_info": "Std-dev",
    }

    class App:
        def __init__(self, root):
            self.root = root
            self.config = Config.load()
            self.event_q: queue.Queue = queue.Queue()
            self.job: Optional[AnalysisJob] = None
            self.results: dict = {}
            self.dataset_path = tk.StringVar()
            self.thumbs = ThumbCache()
            self.embedder = None
            self._quarantine_dir: Optional[Path] = None
            self._undo_stack: list[list[tuple[Path, Path]]] = []
            self._setup_style()
            self._setup_window()
            self._build_ui()
            self._poll()
            threading.Thread(target=self._load_embedder, daemon=True).start()

        def _setup_style(self):
            style = ttk.Style()
            try:
                style.theme_use("clam")
            except tk.TclError:
                pass
            self.root.configure(bg=COLOR_BG)
            base_font = ("Segoe UI", 10)
            style.configure(".", background=COLOR_BG, foreground=COLOR_TEXT, font=base_font)
            style.configure("TFrame", background=COLOR_BG)
            style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT)
            style.configure("TLabelframe", background=COLOR_BG, foreground=COLOR_TEXT)
            style.configure("TLabelframe.Label", background=COLOR_BG, foreground=COLOR_TEXT)
            style.configure("TCheckbutton", background=COLOR_BG, foreground=COLOR_TEXT)
            style.map("TCheckbutton", background=[("active", COLOR_BG)])
            style.configure("Brand.TFrame", background=COLOR_BG)
            style.configure("Brand.TLabel", background=COLOR_BG, foreground=COLOR_NAVY)
            style.configure("BrandTagline.TLabel", background=COLOR_BG,
                            foreground=COLOR_TEXT_MUTED, font=("Segoe UI", 9))
            style.configure("H1.TLabel", font=("Segoe UI", 22, "bold"),
                            background=COLOR_BG, foreground=COLOR_NAVY)
            style.configure("H2.TLabel", font=("Segoe UI", 13, "bold"),
                            background=COLOR_BG, foreground=COLOR_NAVY)
            style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"),
                            background=COLOR_BG, foreground=COLOR_NAVY)
            style.configure("Sub.TLabel", font=("Segoe UI", 9),
                            foreground=COLOR_TEXT_MUTED, background=COLOR_BG)
            style.configure("Pill.TLabel", font=("Segoe UI", 9, "bold"),
                            background=COLOR_NAVY, foreground="#ffffff", padding=(8, 2))
            style.configure("TEntry", fieldbackground=COLOR_SURFACE,
                            bordercolor=COLOR_BORDER, lightcolor=COLOR_BORDER,
                            darkcolor=COLOR_BORDER, padding=6)
            style.configure("TButton", padding=(12, 6), font=("Segoe UI", 10),
                            background=COLOR_SURFACE, foreground=COLOR_NAVY,
                            bordercolor=COLOR_BORDER, focusthickness=0)
            style.map("TButton",
                      background=[("active", "#eef0f7"), ("disabled", "#eef0f7")],
                      foreground=[("disabled", "#9aa0b4")])
            style.configure("Accent.TButton", padding=(16, 8),
                            font=("Segoe UI", 10, "bold"),
                            background=COLOR_MAGENTA, foreground="#ffffff",
                            bordercolor=COLOR_MAGENTA, focusthickness=0)
            style.map("Accent.TButton",
                      background=[("active", COLOR_MAGENTA_HOVER),
                                  ("disabled", "#f2c8d6")],
                      foreground=[("disabled", "#ffffff")])
            style.configure("Primary.TButton", padding=(14, 7),
                            font=("Segoe UI", 10, "bold"),
                            background=COLOR_NAVY, foreground="#ffffff",
                            bordercolor=COLOR_NAVY, focusthickness=0)
            style.map("Primary.TButton",
                      background=[("active", COLOR_NAVY_DEEP),
                                  ("disabled", "#c7cad8")])
            style.configure("Danger.TButton", padding=(14, 7),
                            font=("Segoe UI", 10, "bold"),
                            background=COLOR_DANGER, foreground="#ffffff",
                            bordercolor=COLOR_DANGER, focusthickness=0)
            style.map("Danger.TButton",
                      background=[("active", "#b91c1c"),
                                  ("disabled", "#f1bdbd")])
            style.configure("Card.TFrame", background=COLOR_SURFACE,
                            relief="flat", borderwidth=1, bordercolor=COLOR_BORDER)
            style.configure("Card.TLabel", background=COLOR_SURFACE,
                            foreground=COLOR_TEXT)
            style.configure("CardTitle.TLabel", background=COLOR_SURFACE,
                            foreground=COLOR_NAVY,
                            font=("Segoe UI", 11, "bold"))
            style.configure("CardCount.TLabel", background=COLOR_SURFACE,
                            font=("Segoe UI", 26, "bold"), foreground=COLOR_MAGENTA)
            style.configure("CardBadge.TLabel", background=COLOR_SURFACE,
                            foreground=COLOR_TEXT_MUTED,
                            font=("Segoe UI", 8, "bold"))
            style.configure("TProgressbar", thickness=8,
                            background=COLOR_MAGENTA,
                            troughcolor=COLOR_BORDER,
                            bordercolor=COLOR_BORDER,
                            lightcolor=COLOR_MAGENTA,
                            darkcolor=COLOR_MAGENTA)

        def _setup_window(self):
            self.root.title("Genow Dataset Cleaner")
            self.root.geometry("1080x880")
            self.root.minsize(900, 720)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
            self._logo_header = None
            self._logo_icon = None
            if LOGO_PATH.exists():
                try:
                    with Image.open(LOGO_PATH) as raw:
                        img = raw.convert("RGBA")
                        h_ratio = 56 / img.height
                        header_img = img.resize(
                            (max(1, int(img.width * h_ratio)), 56),
                            Image.LANCZOS,
                        )
                        self._logo_header = ImageTk.PhotoImage(header_img)
                        icon_img = img.resize((64, 64), Image.LANCZOS)
                        self._logo_icon = ImageTk.PhotoImage(icon_img)
                    self.root.iconphoto(True, self._logo_icon)
                except (OSError, ValueError) as e:
                    logger.warning("Logo load failed: %s", e)

        def _build_ui(self):
            brand = tk.Frame(self.root, bg=COLOR_BG, height=88)
            brand.pack(fill="x")
            brand.pack_propagate(False)
            brand_inner = tk.Frame(brand, bg=COLOR_BG)
            brand_inner.pack(fill="both", expand=True, padx=24, pady=12)
            if self._logo_header is not None:
                logo_lbl = tk.Label(brand_inner, image=self._logo_header,
                                    bg=COLOR_BG, bd=0)
                logo_lbl.image = self._logo_header
                logo_lbl.pack(side="left")
            else:
                tk.Label(brand_inner, text="Genow", bg=COLOR_BG, fg=COLOR_NAVY,
                         font=("Segoe UI", 22, "bold")).pack(side="left")
            meta = tk.Frame(brand_inner, bg=COLOR_BG)
            meta.pack(side="right")
            tk.Label(meta, text=f"  {device.upper()}  ",
                     bg=COLOR_MAGENTA, fg="#ffffff",
                     font=("Segoe UI", 9, "bold"), padx=10, pady=3
                     ).pack(side="right", padx=(8, 0))
            trash_note = "system trash" if HAS_SEND2TRASH else "quarantine folder"
            tk.Label(meta, text=f"Deletion: {trash_note}",
                     bg=COLOR_BG, fg=COLOR_TEXT_MUTED,
                     font=("Segoe UI", 9)).pack(side="right")

            title_wrap = ttk.Frame(self.root, padding=(24, 18, 24, 4))
            title_wrap.pack(fill="x")
            ttk.Label(title_wrap, text="Dataset Cleaner", style="H1.TLabel").pack(anchor="w")
            ttk.Label(title_wrap,
                      text="Curate image datasets — detect blur, exposure issues, "
                           "low-information frames, and CLIP-based duplicates.",
                      style="Sub.TLabel").pack(anchor="w", pady=(2, 0))

            pathf = ttk.Frame(self.root, padding=(24, 12, 24, 4))
            pathf.pack(fill="x")
            ttk.Label(pathf, text="Dataset folder", style="H2.TLabel").pack(anchor="w")
            row = ttk.Frame(pathf)
            row.pack(fill="x", pady=(6, 0))
            self.path_entry = ttk.Entry(row, textvariable=self.dataset_path)
            self.path_entry.pack(side="left", fill="x", expand=True, ipady=4)
            ttk.Button(row, text="Browse…", command=self._browse).pack(side="left", padx=(8, 0))

            self._build_settings()

            actions = ttk.Frame(self.root, padding=(24, 10, 24, 4))
            actions.pack(fill="x")
            self.analyze_btn = ttk.Button(actions, text="Analyze dataset",
                                          style="Accent.TButton",
                                          command=self._on_analyze, state="disabled")
            self.analyze_btn.pack(side="left")
            self.cancel_btn = ttk.Button(actions, text="Cancel",
                                         command=self._on_cancel, state="disabled")
            self.cancel_btn.pack(side="left", padx=(8, 0))
            self.undo_btn = ttk.Button(actions, text="Undo last delete",
                                       command=self._on_undo, state="disabled")
            self.undo_btn.pack(side="left", padx=(8, 0))
            self.status_lbl = ttk.Label(actions, text="Loading embedder…",
                                        style="Sub.TLabel")
            self.status_lbl.pack(side="left", padx=14)

            self.progress = ttk.Progressbar(self.root, mode="determinate", maximum=100)
            self.progress.pack(fill="x", padx=24, pady=(4, 8))

            self.results_frame = ttk.Frame(self.root, padding=(20, 4, 20, 4))
            self.results_frame.pack(fill="x")
            self._render_result_cards(empty=True)

            logf = ttk.Frame(self.root, padding=(24, 8, 24, 16))
            logf.pack(fill="both", expand=True)
            ttk.Label(logf, text="Activity log", style="H2.TLabel").pack(anchor="w")
            log_box = tk.Frame(logf, bg=COLOR_BORDER, bd=0, highlightthickness=1,
                               highlightbackground=COLOR_BORDER)
            log_box.pack(fill="both", expand=True, pady=(6, 0))
            self.log = tk.Text(log_box, height=8, wrap="word",
                               bg=COLOR_SURFACE, fg=COLOR_TEXT,
                               relief="flat", borderwidth=0,
                               padx=10, pady=8,
                               font=("Consolas", 9))
            self.log.pack(side="left", fill="both", expand=True)
            sb = ttk.Scrollbar(log_box, orient="vertical", command=self.log.yview)
            sb.pack(side="right", fill="y")
            self.log.configure(yscrollcommand=sb.set, state="disabled")

        def _build_settings(self):
            wrap = ttk.Frame(self.root, padding=(24, 8, 24, 4))
            wrap.pack(fill="x")

            row1 = ttk.Frame(wrap)
            row1.pack(fill="x")
            self.name_filter_var = tk.BooleanVar(value=self.config.enable_name_filter)
            ttk.Checkbutton(row1,
                            text="Flag filenames containing 'det' / 'seg' tokens",
                            variable=self.name_filter_var,
                            command=lambda: self._update_cfg("enable_name_filter",
                                                              self.name_filter_var)
                            ).pack(side="left", padx=4)
            self.trash_var = tk.BooleanVar(value=self.config.use_trash and HAS_SEND2TRASH)
            label = "Use system trash" if HAS_SEND2TRASH else "Use system trash (install 'send2trash')"
            cb = ttk.Checkbutton(row1, text=label, variable=self.trash_var,
                                 command=lambda: self._update_cfg("use_trash",
                                                                   self.trash_var))
            cb.pack(side="left", padx=4)
            if not HAS_SEND2TRASH:
                cb.state(["disabled"])
            self.phash_var = tk.BooleanVar(value=self.config.phash_prepass)
            ph_cb = ttk.Checkbutton(
                row1,
                text="Perceptual-hash pre-pass" + ("" if HAS_IMAGEHASH else " (install 'imagehash')"),
                variable=self.phash_var,
                command=lambda: self._update_cfg("phash_prepass", self.phash_var))
            ph_cb.pack(side="left", padx=4)
            if not HAS_IMAGEHASH:
                ph_cb.state(["disabled"])

            row2 = ttk.Frame(wrap)
            row2.pack(fill="x", pady=(8, 0))

            def _num_field(parent, label_txt, attr, width=8):
                ttk.Label(parent, text=label_txt).pack(side="left", padx=(0, 4))
                var = tk.StringVar(value=str(getattr(self.config, attr)))
                ent = ttk.Entry(parent, textvariable=var, width=width)
                ent.pack(side="left", padx=(0, 12))

                def _commit(*_):
                    try:
                        val = float(var.get())
                    except ValueError:
                        var.set(str(getattr(self.config, attr)))
                        return
                    setattr(self.config, attr, val)
                    self.config.save()

                ent.bind("<FocusOut>", _commit)
                ent.bind("<Return>", _commit)
                return var

            _num_field(row2, "Blur <", "blur_threshold")
            _num_field(row2, "Dark <", "dark_threshold")
            _num_field(row2, "Bright >", "bright_threshold")
            _num_field(row2, "Low-info <", "low_info_threshold")
            _num_field(row2, "Dup ≥", "duplicate_threshold")

            row3 = ttk.Frame(wrap)
            row3.pack(fill="x", pady=(6, 0))

            ttk.Label(row3, text="Embedder (VLM):").pack(side="left", padx=(0, 4))
            embedder_labels = [s.label for s in AVAILABLE_EMBEDDERS]
            embedder_keys = [s.key for s in AVAILABLE_EMBEDDERS]
            current_label = next(
                (s.label for s in AVAILABLE_EMBEDDERS if s.key == self.config.embedder),
                embedder_labels[0],
            )
            self.embedder_label_var = tk.StringVar(value=current_label)
            emb_combo = ttk.Combobox(row3, textvariable=self.embedder_label_var,
                                     values=embedder_labels, width=42, state="readonly")
            emb_combo.pack(side="left", padx=(0, 12))

            def _commit_embedder(*_):
                sel = self.embedder_label_var.get()
                idx = embedder_labels.index(sel) if sel in embedder_labels else 0
                self.config.embedder = embedder_keys[idx]
                self.config.save()
                self.status_lbl.config(
                    text="Embedder change applies on next launch.")

            emb_combo.bind("<<ComboboxSelected>>", _commit_embedder)

            ttk.Button(row3, text="API Keys…",
                       command=self._open_api_keys).pack(side="left", padx=(0, 12))

            ttk.Label(row3, text="Workers:").pack(side="left", padx=(0, 4))
            w_var = tk.StringVar(value=str(self.config.workers))
            w_ent = ttk.Entry(row3, textvariable=w_var, width=4)
            w_ent.pack(side="left")

            def _commit_workers(*_):
                try:
                    self.config.workers = max(1, int(w_var.get()))
                except ValueError:
                    w_var.set(str(self.config.workers))
                    return
                self.config.save()

            w_ent.bind("<FocusOut>", _commit_workers)
            w_ent.bind("<Return>", _commit_workers)

        def _update_cfg(self, key, var):
            try:
                val = var.get()
            except tk.TclError:
                return
            setattr(self.config, key, val)
            self.config.save()

        def _render_result_cards(self, empty=False):
            for w in self.results_frame.winfo_children():
                w.destroy()
            ttk.Label(self.results_frame, text="Results", style="H2.TLabel").pack(
                anchor="w", padx=4, pady=(0, 6))
            grid = ttk.Frame(self.results_frame)
            grid.pack(fill="x")
            cols = 4
            for idx, (key, label, badge) in enumerate(CATEGORY_META):
                r, c = divmod(idx, cols)
                grid.columnconfigure(c, weight=1)
                count = self._count(key)
                active = count > 0 and not empty
                card = ttk.Frame(grid, style="Card.TFrame", padding=14)
                card.grid(row=r, column=c, sticky="nsew", padx=6, pady=6)
                top = ttk.Frame(card, style="Card.TFrame")
                top.pack(fill="x")
                ttk.Label(top, text=badge, style="CardBadge.TLabel").pack(side="left")
                ttk.Label(top, text=label, style="CardTitle.TLabel").pack(
                    side="left", padx=8)
                count_color = COLOR_MAGENTA if active else "#c7cad8"
                count_lbl = ttk.Label(card, text=str(count),
                                      style="CardCount.TLabel")
                count_lbl.configure(foreground=count_color)
                count_lbl.pack(anchor="w", pady=(6, 8))
                btn_row = ttk.Frame(card, style="Card.TFrame")
                btn_row.pack(anchor="w", fill="x")
                ttk.Button(btn_row, text="Review",
                           command=lambda k=key: self._open_review(k),
                           state=("normal" if active else "disabled")
                           ).pack(side="left")
                if key == "duplicates":
                    ttk.Button(btn_row, text="Auto-clean",
                               style="Danger.TButton",
                               command=self._auto_clean_duplicates,
                               state=("normal" if active else "disabled")
                               ).pack(side="left", padx=(6, 0))

        def _count(self, key):
            if not self.results:
                return 0
            if key == "duplicates":
                return sum(max(0, len(c) - 1)
                           for c in self.results.get("duplicate_clusters", []))
            return len(self.results.get(key, []))

        def _auto_clean_duplicates(self):
            clusters = self.results.get("duplicate_clusters", [])
            if not clusters:
                return
            to_delete: list[Path] = []
            for members in clusters:
                to_delete.extend(members[1:])
            if not to_delete:
                return
            if not messagebox.askyesno(
                    "Auto-clean duplicates",
                    f"Keep the highest-quality image in each of {len(clusters)} cluster(s) "
                    f"and remove {len(to_delete)} duplicate(s)?\n\n"
                    f"Quality = sharpness + info + resolution − brightness extremes.\n"
                    f"Companion label files will be removed as well."):
                return
            self._delete_paths(to_delete)

        def _open_review(self, key):
            if key == "duplicates":
                clusters = self.results.get("duplicate_clusters", [])
                if not clusters:
                    return
                DuplicateReview(self.root, clusters,
                                self.results.get("quality_scores", {}),
                                self._delete_paths,
                                self._auto_clean_duplicates, self.thumbs)
                return
            items = self.results.get(key, [])
            if not items:
                return
            title = dict((k, lbl) for k, lbl, _ in CATEGORY_META)[key]
            CategoryReview(self.root, title, items, SCORE_LABELS.get(key, ""),
                           self._delete_paths, self.thumbs)

        def _ensure_quarantine(self) -> Path:
            if self._quarantine_dir is None:
                ds_name = Path(self.dataset_path.get()).name or "dataset"
                self._quarantine_dir = (APP_DIR / "quarantine" /
                                        f"{ds_name}_{datetime.now():%Y%m%d_%H%M%S}")
            return self._quarantine_dir

        def _delete_paths(self, paths):
            use_trash = bool(self.trash_var.get() and HAS_SEND2TRASH)
            quarantine = self._ensure_quarantine()
            total = 0
            batch_moves: list[tuple[Path, Path]] = []
            for p in paths:
                count, moves = remove_yolo_pair(p, use_trash, quarantine)
                total += count
                batch_moves.extend(moves)
            if batch_moves:
                self._undo_stack.append(batch_moves)
                self.undo_btn.state(["!disabled"])
            target = "system trash" if use_trash else f"quarantine: {quarantine}"
            self._append_log(f"Removed {total} file(s) → {target}")

            deleted = set(paths)
            for k in ("blurry", "dark", "bright", "low_info"):
                self.results[k] = [it for it in self.results.get(k, [])
                                   if (it[0] if isinstance(it, tuple) else it) not in deleted]
            for k in ("name_filtered", "unreadable"):
                self.results[k] = [it for it in self.results.get(k, []) if it not in deleted]
            new_clusters = []
            for c in self.results.get("duplicate_clusters", []):
                remaining = [p for p in c if p not in deleted]
                if len(remaining) >= 2:
                    new_clusters.append(remaining)
            self.results["duplicate_clusters"] = new_clusters
            self._render_result_cards()
            return total

        def _on_undo(self):
            if not self._undo_stack:
                return
            batch = self._undo_stack.pop()
            restored = 0
            failed = 0
            for original, moved_to in batch:
                if not moved_to.exists():
                    failed += 1
                    continue
                try:
                    original.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(moved_to), original)
                    restored += 1
                except OSError as e:
                    logger.error("Restore failed for %s: %s", moved_to, e)
                    failed += 1
            msg = f"Restored {restored} file(s)"
            if failed:
                msg += f" ({failed} failed)"
            self._append_log(msg)
            if not self._undo_stack:
                self.undo_btn.state(["disabled"])

        def _browse(self):
            p = filedialog.askdirectory()
            if p:
                self.dataset_path.set(p)

        def _open_api_keys(self):
            from api_embedders import KEY_ENV_VARS, KeyStore
            win = tk.Toplevel(self.root)
            win.title("API Keys")
            win.configure(bg=COLOR_BG)
            win.geometry("560x360")
            ttk.Label(win, text="API keys for hosted VLM embedders",
                      style="Header.TLabel").pack(anchor="w", padx=16, pady=(14, 2))
            ttk.Label(win,
                      text="Stored in ~/.genow_cleaner/api_keys.json (chmod 600). "
                           "Environment variables, if set, take precedence.",
                      style="Sub.TLabel", wraplength=520).pack(anchor="w", padx=16)

            body = ttk.Frame(win, padding=16)
            body.pack(fill="both", expand=True)
            entries: dict[str, tk.StringVar] = {}
            for r, (family, env_var) in enumerate(KEY_ENV_VARS.items()):
                ttk.Label(body, text=family).grid(row=r, column=0, sticky="w", pady=4)
                var = tk.StringVar()
                try:
                    existing = KeyStore.get(family)
                    var.set(existing)
                except RuntimeError:
                    pass
                ent = ttk.Entry(body, textvariable=var, width=44, show="•")
                ent.grid(row=r, column=1, padx=8, pady=4)
                ttk.Label(body, text=f"env: {env_var}", style="Sub.TLabel").grid(
                    row=r, column=2, sticky="w")
                entries[family] = var
            body.columnconfigure(1, weight=1)

            def _save():
                for family, var in entries.items():
                    KeyStore.set(family, var.get().strip())
                messagebox.showinfo("Saved", "API keys saved.")
                win.destroy()

            footer = ttk.Frame(win, padding=12)
            footer.pack(fill="x")
            ttk.Button(footer, text="Cancel", command=win.destroy).pack(side="right", padx=4)
            ttk.Button(footer, text="Save", style="Primary.TButton",
                       command=_save).pack(side="right", padx=4)

        def _on_analyze(self):
            raw = self.dataset_path.get().strip()
            path = Path(raw) if raw else None
            if not path or not path.is_dir():
                messagebox.showerror("Invalid path", "Please choose a valid dataset folder.")
                return
            ok, reason = validate_yolo_dataset(path)
            if not ok:
                messagebox.showerror("Unsupported dataset", reason)
                return
            if self.embedder is None:
                messagebox.showinfo("Please wait", "Embedder is still loading.")
                return
            self._quarantine_dir = None
            self._undo_stack.clear()
            self.undo_btn.state(["disabled"])
            self.results = {}
            self._render_result_cards(empty=True)
            self.progress["value"] = 0
            self.status_lbl.config(text="Analyzing…")
            self.analyze_btn.state(["disabled"])
            self.cancel_btn.state(["!disabled"])
            self._append_log(f"Starting analysis: {path} ({reason})")
            self.job = AnalysisJob(path, self.config, self.event_q, self.embedder)
            self.job.start()

        def _on_cancel(self):
            if self.job and self.job.is_alive():
                self.job.cancel()
                self._append_log("Cancelling…")
                self.cancel_btn.state(["disabled"])

        def _on_close(self):
            if self.job and self.job.is_alive():
                if not messagebox.askyesno("Quit",
                                           "Analysis is running. Cancel and quit?"):
                    return
                self.job.cancel()
            self.root.destroy()

        def _append_log(self, message):
            self.log.configure(state="normal")
            self.log.insert("end", f"{datetime.now():%H:%M:%S}  {message}\n")
            self.log.see("end")
            self.log.configure(state="disabled")

        def _poll(self):
            try:
                while True:
                    evt = self.event_q.get_nowait()
                    self._handle_event(evt)
            except queue.Empty:
                pass
            self.root.after(80, self._poll)

        def _handle_event(self, evt):
            if evt.kind == "progress":
                self.progress["value"] = evt.value
                if evt.message:
                    self.status_lbl.config(text=evt.message)
            elif evt.kind == "log":
                self._append_log(evt.message)
            elif evt.kind == "done":
                self.results = evt.payload or {}
                self.progress["value"] = 100
                counts = {k: self._count(k) for k, _, _ in CATEGORY_META}
                self.status_lbl.config(text="Analysis complete.")
                self._append_log("Analysis complete: " +
                                 ", ".join(f"{k}={v}" for k, v in counts.items()))
                self._render_result_cards()
                # Auto-write report
                try:
                    ds_name = Path(self.dataset_path.get()).name or "dataset"
                    report_dir = APP_DIR / "reports"
                    report_path = (report_dir /
                                   f"{ds_name}_{datetime.now():%Y%m%d_%H%M%S}.json")
                    write_report(report_path,
                                 Path(self.dataset_path.get()),
                                 self.results, self.config)
                    self._append_log(f"Report saved: {report_path}")
                except OSError as e:
                    self._append_log(f"Report save failed: {e}")
                self.analyze_btn.state(["!disabled"])
                self.cancel_btn.state(["disabled"])
            elif evt.kind == "cancelled":
                self.status_lbl.config(text="Cancelled.")
                self._append_log("Analysis cancelled.")
                self.progress["value"] = 0
                self.analyze_btn.state(["!disabled"])
                self.cancel_btn.state(["disabled"])
            elif evt.kind == "error":
                self.status_lbl.config(text="Error.")
                self._append_log(f"ERROR: {evt.message}")
                messagebox.showerror("Analysis error", evt.message)
                self.analyze_btn.state(["!disabled"])
                self.cancel_btn.state(["disabled"])
            elif evt.kind == "embedder_ready":
                self.embedder = evt.payload
                spec = self.embedder.spec
                self.status_lbl.config(
                    text=f"Ready. {spec.label} loaded on {device.upper()}.")
                self.analyze_btn.state(["!disabled"])
            elif evt.kind == "embedder_error":
                self.status_lbl.config(text="Embedder load failed.")
                self._append_log(f"Embedder load failed: {evt.message}")
                messagebox.showerror("Embedder error", evt.message)

        def _load_embedder(self):
            try:
                emb = load_embedder(self.config.embedder, device)
                self.event_q.put(Event(kind="embedder_ready", payload=emb))
            except Exception as e:
                self.event_q.put(Event(kind="embedder_error", message=str(e)))

    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


# ---------- CLI ----------

def _cli_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="genowCleaner",
                                 description="Headless dataset analysis")
    p.add_argument("dataset", type=Path, help="Path to dataset directory")
    p.add_argument("--report", type=Path, default=None,
                   help="Where to write JSON report (default: ~/.genow_cleaner/reports/)")
    p.add_argument("--auto-clean-duplicates", action="store_true",
                   help="Keep best-quality image per cluster, remove others")
    p.add_argument("--blur-threshold", type=float)
    p.add_argument("--dark-threshold", type=float)
    p.add_argument("--bright-threshold", type=float)
    p.add_argument("--low-info-threshold", type=float)
    p.add_argument("--duplicate-threshold", type=float)
    p.add_argument("--embedder", type=str,
                   choices=[s.key for s in AVAILABLE_EMBEDDERS],
                   help="Image embedder/VLM key")
    p.add_argument("--workers", type=int)
    p.add_argument("--no-phash", action="store_true",
                   help="Disable perceptual-hash pre-pass")
    p.add_argument("--use-trash", action="store_true",
                   help="Use system trash (if available) instead of quarantine")
    args = p.parse_args(argv)

    ok, reason = validate_yolo_dataset(args.dataset)
    if not ok:
        print(reason, file=sys.stderr)
        return 2
    print(f"YOLOv11/v12 dataset OK ({reason}).")

    config = Config.load()
    for attr in ("blur_threshold", "dark_threshold", "bright_threshold",
                 "low_info_threshold", "duplicate_threshold", "embedder",
                 "workers"):
        val = getattr(args, attr, None)
        if val is not None:
            setattr(config, attr, val)
    if args.no_phash:
        config.phash_prepass = False

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = get_spec(config.embedder)
    print(f"Loading embedder {spec.label} on {device}…")
    embedder = load_embedder(config.embedder, device)

    event_q: queue.Queue = queue.Queue()
    job = AnalysisJob(args.dataset, config, event_q, embedder)
    # Run synchronously in-process (skip thread.start to print events live)
    last_progress = 0.0
    job_thread = threading.Thread(target=job.run, daemon=True)
    job_thread.start()
    results: dict = {}
    while True:
        try:
            evt = event_q.get(timeout=0.5)
        except queue.Empty:
            if not job_thread.is_alive():
                break
            continue
        if evt.kind == "log":
            print(evt.message)
        elif evt.kind == "progress":
            if evt.value - last_progress >= 5:
                last_progress = evt.value
                print(f"  [{evt.value:5.1f}%] {evt.message}")
        elif evt.kind == "done":
            results = evt.payload or {}
            print("Analysis complete.")
        elif evt.kind == "error":
            print(f"ERROR: {evt.message}", file=sys.stderr)
            return 1
        elif evt.kind == "cancelled":
            print("Cancelled.")
            return 130

    if args.report is None:
        ds_name = args.dataset.name or "dataset"
        args.report = (APP_DIR / "reports" /
                       f"{ds_name}_{datetime.now():%Y%m%d_%H%M%S}.json")
    write_report(args.report, args.dataset, results, config)
    print(f"Report: {args.report}")

    n_dup = sum(max(0, len(c) - 1)
                for c in results.get("duplicate_clusters", []))
    summary = {
        "scanned": results.get("total_scanned", 0),
        "blurry": len(results.get("blurry", [])),
        "dark": len(results.get("dark", [])),
        "bright": len(results.get("bright", [])),
        "low_info": len(results.get("low_info", [])),
        "name_filtered": len(results.get("name_filtered", [])),
        "unreadable": len(results.get("unreadable", [])),
        "duplicates": n_dup,
    }
    print("Summary:", json.dumps(summary, indent=2))

    if args.auto_clean_duplicates:
        clusters = results.get("duplicate_clusters", [])
        to_delete: list[Path] = []
        for members in clusters:
            to_delete.extend(members[1:])
        if not to_delete:
            print("No duplicates to clean.")
        else:
            use_trash = args.use_trash and HAS_SEND2TRASH
            quarantine = (APP_DIR / "quarantine" /
                          f"{args.dataset.name}_{datetime.now():%Y%m%d_%H%M%S}")
            total = 0
            for path in to_delete:
                count, _ = remove_yolo_pair(path, use_trash, quarantine)
                total += count
            target = "system trash" if use_trash else f"quarantine: {quarantine}"
            print(f"Auto-cleaned {total} file(s) → {target}")

    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in ("--cli", "cli"):
        return _cli_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    return _run_gui()


if __name__ == "__main__":
    sys.exit(main())
