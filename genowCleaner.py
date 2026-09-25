#!/usr/bin/env python3
"""GenowDatasetCleaner — image dataset cleaner with VLM-based dedup.

Works on any folder of images. YOLOv11 / YOLOv12 (Ultralytics) datasets get
extra handling: a folder containing a `data.yaml` with `train:` + `names:`/`nc:`,
or parallel `images/` and `labels/` directories, is treated as YOLO, and removing
an image also removes its companion label `.txt`. Any other folder is processed
as a plain image collection (images only, labels untouched).

Usage:
    python genowCleaner.py                 # launch GUI
    python genowCleaner.py --cli DATASET   # headless analysis, writes JSON report
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import queue
import re
import shutil
import subprocess
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
def _resource_dir() -> Path:
    # When frozen by PyInstaller, bundled data lives under sys._MEIPASS.
    return Path(getattr(sys, "_MEIPASS", Path(__file__).parent))

LOGO_PATH = _resource_dir() / "genow_logo.jpeg"

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


def collect_images(dataset_path: Path, skip_labels: bool = True) -> list[Path]:
    """Recursively collect image files without following symlinks.

    With `skip_labels`, skips the YOLO `labels/` subtrees so label-adjacent
    files don't get scanned as images. Always skips quarantine/report dirs.
    """
    out: list[Path] = []
    for root, dirs, files in os.walk(dataset_path, followlinks=False):
        dirs[:] = [d for d in dirs
                   if not d.startswith("_quarantine_")
                   and not d.startswith(".genow_cleaner")
                   and not (skip_labels and d.lower() == "labels")]
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


FORMAT_YOLO = "yolo"
FORMAT_IMAGES = "images"
FORMAT_NAMES = {FORMAT_YOLO: "YOLOv11/v12", FORMAT_IMAGES: "image folder"}


def detect_dataset_format(path: Path) -> tuple[Optional[str], str]:
    """Return (format, reason). format is FORMAT_YOLO for a YOLO dataset,
    FORMAT_IMAGES for any other directory, or None if path is unusable."""
    if not path.is_dir():
        return None, f"Not a directory: {path}"
    ok, reason = validate_yolo_dataset(path)
    if ok:
        return FORMAT_YOLO, reason
    return FORMAT_IMAGES, "no YOLO layout found, treating as a plain image folder"


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


SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val",
                 "validation": "val", "test": "test"}
# Lower rank = kept first. Evaluation splits are small and often fixed, so a
# duplicate that leaks across splits is removed from train, not from val/test.
SPLIT_KEEP_RANK = {"test": 0, "val": 1, "train": 2}


def yolo_split(image_path: Path) -> Optional[str]:
    """Return the canonical split ("train"/"val"/"test") of an image, or None.

    Handles both `images/<split>/foo.jpg` (Ultralytics) and
    `<split>/images/foo.jpg` (Roboflow export) layouts.
    """
    for part in reversed(image_path.parts[:-1]):
        split = SPLIT_ALIASES.get(part.lower())
        if split:
            return split
    return None


def keep_priority(path: Path, quality: dict[Path, dict]) -> tuple[int, float]:
    """Sort key for duplicate members: first element is the one to keep."""
    rank = SPLIT_KEEP_RANK.get(yolo_split(path), len(SPLIT_KEEP_RANK))
    return rank, -quality_score(quality.get(path, {}))


def cluster_splits(members: list[Path]) -> list[str]:
    """Distinct known splits in a cluster, in train/val/test order."""
    found = {yolo_split(p) for p in members} - {None}
    return [sp for sp in ("train", "val", "test") if sp in found]


def cluster_by_similarity(embeddings: np.ndarray, threshold: float,
                          order: Optional[list[int]] = None,
                          cancel_check: Callable[[], bool] = lambda: False,
                          progress: Optional[Callable[[int, int], None]] = None
                          ) -> list[list[int]]:
    """Cluster row-normalized embeddings by cosine similarity > threshold.

    Greedy leader clustering: rows are visited in `order` (best-to-keep first);
    each still-unassigned row becomes a leader and absorbs every unassigned row
    similar to *it*. Unlike transitive linking, this can't chain A~B~C into one
    cluster when A and C differ, so every member is a duplicate of the leader,
    which is returned first.

    Memory is O(chunk * N), not O(N^2). Time is O(N^2).
    """
    n = embeddings.shape[0]
    if n < 2:
        return []
    visit = np.arange(n) if order is None else np.asarray(order, dtype=np.int64)
    assigned = np.zeros(n, dtype=bool)
    clusters: list[list[int]] = []
    chunk = 512
    for s in range(0, n, chunk):
        if cancel_check():
            return []
        rows = visit[s:s + chunk]
        sim = embeddings[rows] @ embeddings.T
        for li, i in enumerate(rows):
            if assigned[i]:
                continue
            assigned[i] = True
            hits = np.where((sim[li] > threshold) & ~assigned)[0]
            if hits.size:
                assigned[hits] = True
                clusters.append([int(i)] + hits.tolist())
        if progress:
            progress(min(n, s + chunk), n)
    return clusters


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
        fmt, reason = detect_dataset_format(self.dataset_path)
        if fmt is None:
            raise RuntimeError(reason)
        self._post("log", message=f"{FORMAT_NAMES[fmt]} dataset ({reason}).")
        self._post("log", message="Collecting image files…")
        all_imgs = collect_images(self.dataset_path,
                                  skip_labels=(fmt == FORMAT_YOLO))
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
                        members.sort(key=lambda p: keep_priority(p, quality))
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
                order=sorted(range(len(valid)),
                             key=lambda k: keep_priority(valid[k], quality)),
                cancel_check=self.cancelled,
                progress=lambda done, total: self._post(
                    "progress", value=90 + done / max(total, 1) * 10,
                    message=f"Dedup {done}/{total}"),
            )
            clip_clusters: list[list[Path]] = []
            for members in clip_clusters_idx:
                ps = [valid[m] for m in members]
                ps.sort(key=lambda p: keep_priority(p, quality))
                clip_clusters.append(ps)
            clusters = phash_dupes + clip_clusters
            clusters.sort(key=len, reverse=True)
            n_cross = sum(1 for c in clusters if len(cluster_splits(c)) > 1)
            if n_cross:
                self._post("log",
                           message=f"{n_cross} duplicate cluster(s) span several splits "
                                   f"(train/val/test leakage); val/test copies are kept.")

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
            "dataset_format": fmt,
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


def remove_yolo_pair(image_path: Path, use_trash: bool, quarantine_dir: Path,
                     with_label: bool = True
                     ) -> tuple[int, list[tuple[Path, Path]]]:
    """Remove an image and, if `with_label`, its companion YOLO label .txt.

    Returns (count_removed, undo_moves). undo_moves is populated only when files
    were moved to quarantine (system-trash deletes are not tracked for undo).
    """
    moves: list[tuple[Path, Path]] = []
    count = 0
    targets = [image_path]
    if with_label:
        targets.append(yolo_label_path(image_path))
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
            {"path": str(p), "split": yolo_split(p),
             "quality_score": scores.get(str(p))} for p in c
        ])

    report = {
        "schema": 2,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "dataset_format": FORMAT_NAMES[results.get("dataset_format", FORMAT_YOLO)],
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
        def __init__(self, master, fill_height=False, **kw):
            """With `fill_height`, the inner frame is stretched to at least the
            visible height so expanding children still fill the window."""
            super().__init__(master, **kw)
            self.fill_height = fill_height
            self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg=COLOR_BG)
            self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=self.vsb.set)
            self.vsb.pack(side="right", fill="y")
            self.canvas.pack(side="left", fill="both", expand=True)
            self.inner = ttk.Frame(self.canvas)
            self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
            self.inner.bind("<Configure>",
                            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
            self.canvas.bind("<Configure>", self._on_canvas_configure)
            if fill_height:
                self.inner.bind("<Configure>", self._on_canvas_configure, add="+")
            self.canvas.bind("<Enter>", self._bind_wheel)
            self.canvas.bind("<Leave>", self._unbind_wheel)

        def _on_canvas_configure(self, _e=None):
            opts = {"width": self.canvas.winfo_width()}
            if self.fill_height:
                opts["height"] = max(self.canvas.winfo_height(),
                                     self.inner.winfo_reqheight())
            self.canvas.itemconfigure(self.inner_id, **opts)

        def _on_wheel(self, evt):
            if isinstance(evt.widget, tk.Text):
                return  # the Text widget scrolls itself
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

    # ----- Full-size viewer -----
    def open_full_image(master, paths, index=0, is_marked=None, on_toggle=None):
        """Show paths[index] fitted to the screen.

        Esc or a click closes, ←/→ step through `paths`, and Del toggles the
        image's delete mark when `on_toggle(i)` is given.
        """
        win = tk.Toplevel(master)
        win.configure(bg=COLOR_NAVY_DEEP)
        win.transient(master)
        max_w = int(win.winfo_screenwidth() * 0.85)
        max_h = int(win.winfo_screenheight() * 0.78)
        img_lbl = tk.Label(win, bg=COLOR_NAVY_DEEP, fg="#ffffff", cursor="hand2")
        img_lbl.pack(padx=12, pady=(12, 6))
        info = tk.Label(win, bg=COLOR_NAVY_DEEP, fg="#ffffff",
                        font=("Segoe UI", 10), wraplength=max_w)
        info.pack(padx=12)
        hint = "Esc/click close  ·  ←/→ previous/next"
        if on_toggle:
            hint += "  ·  Del toggle delete"
        tk.Label(win, text=hint, bg=COLOR_NAVY_DEEP, fg="#9aa0b4",
                 font=("Segoe UI", 9)).pack(pady=(2, 10))
        pos = [index]

        def show():
            path = paths[pos[0]]
            win.title(f"{path.name} — {pos[0] + 1}/{len(paths)}")
            try:
                with Image.open(path) as raw:
                    size_txt = f"{raw.width}×{raw.height}"
                    img = raw.convert("RGB")
                img.thumbnail((max_w, max_h))
                photo = ImageTk.PhotoImage(img)
                img_lbl.configure(image=photo, text="")
                img_lbl.image = photo
            except (OSError, ValueError):
                img_lbl.configure(image="", text="(unreadable)")
                img_lbl.image = None
                size_txt = "unreadable"
            text = f"{path}   ·   {size_txt}"
            if is_marked and is_marked(pos[0]):
                text += "   ·   MARKED FOR DELETION"
            info.configure(text=text)

        def step(delta):
            pos[0] = (pos[0] + delta) % len(paths)
            show()

        def toggle():
            on_toggle(pos[0])
            show()

        win.bind("<Escape>", lambda _e: win.destroy())
        img_lbl.bind("<Button-1>", lambda _e: win.destroy())
        win.bind("<Left>", lambda _e: step(-1))
        win.bind("<Right>", lambda _e: step(1))
        if on_toggle:
            win.bind("<Delete>", lambda _e: toggle())
        show()
        win.focus_set()
        return win

    # ----- Review windows -----
    class PagedReview(tk.Toplevel):
        """Paginated review window with shared keyboard shortcuts.

        Esc close, ←/→ page, Del delete, Ctrl+A / Ctrl+D select all / none.
        Subclasses implement _render_page, _set_all and _confirm.
        """
        page_size = 50

        def __init__(self, master, n_units, geometry, title, subtitle, thumb_cache):
            super().__init__(master)
            self.geometry(geometry)
            self.configure(bg=COLOR_BG)
            self.thumbs = thumb_cache
            self.page = 0
            self.n_pages = max(1, -(-n_units // self.page_size))

            hdr = ttk.Frame(self, padding=12)
            hdr.pack(fill="x")
            ttk.Label(hdr, text=title, style="Header.TLabel").pack(side="left")
            ttk.Label(hdr, text=f"   {subtitle}", style="Sub.TLabel").pack(side="left")

            self.toolbar = ttk.Frame(self, padding=(12, 0))
            self.toolbar.pack(fill="x")
            pager = ttk.Frame(self.toolbar)
            pager.pack(side="right")
            self.prev_btn = ttk.Button(pager, text="◀ Prev",
                                       command=lambda: self._goto(self.page - 1))
            self.prev_btn.pack(side="left")
            self.page_lbl = ttk.Label(pager, style="Sub.TLabel")
            self.page_lbl.pack(side="left", padx=8)
            self.next_btn = ttk.Button(pager, text="Next ▶",
                                       command=lambda: self._goto(self.page + 1))
            self.next_btn.pack(side="left")

            self.body = ScrollableFrame(self)
            self.body.pack(fill="both", expand=True, padx=12, pady=10)

            self.footer = ttk.Frame(self, padding=12)
            self.footer.pack(fill="x")
            ttk.Label(self.footer,
                      text="Esc close · ←/→ page · Del delete · Ctrl+A / Ctrl+D "
                           "select all / none · click a thumbnail to zoom",
                      style="Sub.TLabel").pack(side="left")

            self.bind("<Escape>", lambda _e: self.destroy())
            self.bind("<Left>", lambda _e: self._goto(self.page - 1))
            self.bind("<Right>", lambda _e: self._goto(self.page + 1))
            self.bind("<Delete>", lambda _e: self._confirm())
            for seq in ("<Control-a>", "<Control-A>"):
                self.bind(seq, lambda _e: self._set_all(True) or "break")
            for seq in ("<Control-d>", "<Control-D>"):
                self.bind(seq, lambda _e: self._set_all(False) or "break")
            self.focus_set()

        def _goto(self, page):
            if not 0 <= page < self.n_pages:
                return
            self.page = page
            for w in self.body.inner.winfo_children():
                w.destroy()
            self._render_page()
            self.body.canvas.yview_moveto(0)
            self.page_lbl.config(text=f"Page {page + 1}/{self.n_pages}")
            self.prev_btn.state(["!disabled" if page > 0 else "disabled"])
            self.next_btn.state(["!disabled" if page < self.n_pages - 1 else "disabled"])

        def _page_range(self, n):
            start = self.page * self.page_size
            return range(start, min(n, start + self.page_size))

        def _thumb(self, parent, path, on_click):
            photo = self.thumbs.get(path)
            if photo:
                lbl = ttk.Label(parent, image=photo, style="Card.TLabel", cursor="hand2")
                lbl.image = photo
                lbl.bind("<Button-1>", lambda _e: on_click())
            else:
                lbl = ttk.Label(parent, text="(unreadable)", style="Card.TLabel")
            return lbl

        def _render_page(self):
            raise NotImplementedError

        def _set_all(self, val):
            raise NotImplementedError

        def _confirm(self):
            raise NotImplementedError

    class CategoryReview(PagedReview):
        page_size = 50
        cols = 5

        def __init__(self, master, title, items, score_label, on_delete, thumb_cache):
            super().__init__(master, len(items), "1000x760", title,
                             f"{len(items)} item(s)", thumb_cache)
            self.title(f"Review — {title}")
            self.items = items
            self.paths = [it[0] if isinstance(it, tuple) else it for it in items]
            self.score_label = score_label
            self.on_delete = on_delete
            self.selected = [False] * len(items)
            self.page_vars: dict[int, tk.BooleanVar] = {}

            ttk.Button(self.toolbar, text="Select All",
                       command=lambda: self._set_all(True)).pack(side="left", padx=2)
            ttk.Button(self.toolbar, text="Select None",
                       command=lambda: self._set_all(False)).pack(side="left", padx=2)
            self.summary_lbl = ttk.Label(self.toolbar, text="", style="Sub.TLabel")
            self.summary_lbl.pack(side="left", padx=12)

            ttk.Button(self.footer, text="Close", command=self.destroy).pack(side="right", padx=4)
            ttk.Button(self.footer, text="Delete Selected", style="Danger.TButton",
                       command=self._confirm).pack(side="right", padx=4)
            self._update_summary()
            self._goto(0)

        def _render_page(self):
            grid = self.body.inner
            self.page_vars.clear()
            for slot, i in enumerate(self._page_range(len(self.items))):
                item, path = self.items[i], self.paths[i]
                caption = (f"{self.score_label}: {item[1]:.2f}"
                           if isinstance(item, tuple) else "")
                card = ttk.Frame(grid, style="Card.TFrame", padding=6)
                card.grid(row=slot // self.cols, column=slot % self.cols,
                          padx=6, pady=6, sticky="nsew")
                grid.columnconfigure(slot % self.cols, weight=1)
                self._thumb(card, path, lambda i=i: self._zoom(i)).pack()
                ttk.Label(card, text=path.name, style="Card.TLabel",
                          wraplength=170).pack(pady=(4, 0))
                if caption:
                    ttk.Label(card, text=caption, style="Card.TLabel",
                              foreground=COLOR_TEXT_MUTED).pack()
                var = tk.BooleanVar(value=self.selected[i])
                ttk.Checkbutton(card, text="Delete", variable=var,
                                command=lambda i=i, v=var: self._mark(i, v.get())
                                ).pack(pady=(4, 0))
                self.page_vars[i] = var

        def _zoom(self, i):
            open_full_image(self, self.paths, i,
                            is_marked=lambda k: self.selected[k],
                            on_toggle=lambda k: self._mark(k, not self.selected[k]))

        def _mark(self, i, val):
            self.selected[i] = val
            var = self.page_vars.get(i)
            if var is not None and var.get() != val:
                var.set(val)
            self._update_summary()

        def _set_all(self, val):
            self.selected = [val] * len(self.items)
            for var in self.page_vars.values():
                var.set(val)
            self._update_summary()

        def _update_summary(self):
            self.summary_lbl.config(
                text=f"{sum(self.selected)}/{len(self.selected)} selected for deletion")

        def _confirm(self):
            to_delete = [p for p, sel in zip(self.paths, self.selected) if sel]
            if not to_delete:
                messagebox.showinfo("Nothing selected", "No items selected.", parent=self)
                return
            if not messagebox.askyesno(
                    "Confirm deletion",
                    f"Remove {len(to_delete)} item(s)?\n"
                    f"For YOLO datasets, companion label files (.txt) are removed as well.",
                    parent=self):
                return
            deleted = self.on_delete(to_delete)
            messagebox.showinfo("Done", f"{deleted} file(s) removed.", parent=self)
            self.destroy()

    class DuplicateReview(PagedReview):
        page_size = 15
        cols = 6

        def __init__(self, master, clusters, quality_scores, on_delete,
                     on_auto_clean, thumb_cache):
            total_dupes = sum(len(c) - 1 for c in clusters)
            super().__init__(master, len(clusters), "1120x820", "Duplicate clusters",
                             f"{len(clusters)} cluster(s) · {total_dupes} duplicate(s)",
                             thumb_cache)
            self.title("Review — Duplicate clusters")
            self.clusters = clusters
            self.scores = quality_scores
            self.on_delete = on_delete
            self.on_auto_clean = on_auto_clean
            # Indices kept in each cluster; several may be kept, never none.
            self.keep: list[set[int]] = [{0} for _ in clusters]
            self.skip = [False] * len(clusters)
            self.skip_vars: dict[int, tk.BooleanVar] = {}

            ttk.Button(self.toolbar, text="Include All",
                       command=lambda: self._set_all(True)).pack(side="left", padx=2)
            ttk.Button(self.toolbar, text="Skip All",
                       command=lambda: self._set_all(False)).pack(side="left", padx=2)
            self.summary_lbl = ttk.Label(self.toolbar, text="", style="Sub.TLabel")
            self.summary_lbl.pack(side="left", padx=12)

            ttk.Button(self.footer, text="Close", command=self.destroy).pack(side="right", padx=4)
            ttk.Button(self.footer, text="Delete Non-Kept", style="Danger.TButton",
                       command=self._confirm).pack(side="right", padx=4)
            ttk.Button(self.footer, text="Auto-clean (keep best in each)",
                       style="Accent.TButton",
                       command=self._auto_clean).pack(side="right", padx=4)
            self._update_summary()
            self._goto(0)

        def _render_page(self):
            container = self.body.inner
            self.skip_vars.clear()
            for ci in self._page_range(len(self.clusters)):
                members = self.clusters[ci]
                box = ttk.Frame(container, style="Card.TFrame", padding=10)
                box.pack(fill="x", padx=4, pady=6)

                head = ttk.Frame(box, style="Card.TFrame")
                head.pack(fill="x")
                title = f"Cluster {ci + 1} — {len(members)} images"
                splits = cluster_splits(members)
                if len(splits) > 1:
                    title += f"  ·  spans {'/'.join(splits)}"
                ttk.Label(head, text=title,
                          style="CardTitle.TLabel").pack(side="left")
                skip_var = tk.BooleanVar(value=self.skip[ci])
                ttk.Checkbutton(head, text="Skip this cluster (keep all)",
                                variable=skip_var,
                                command=lambda ci=ci, v=skip_var: self._set_skip(ci, v.get())
                                ).pack(side="right")
                self.skip_vars[ci] = skip_var

                grid = ttk.Frame(box, style="Card.TFrame")
                grid.pack(fill="x", pady=(8, 0))
                for mi, path in enumerate(members):
                    card = ttk.Frame(grid, style="Card.TFrame", padding=4)
                    card.grid(row=mi // self.cols, column=mi % self.cols, padx=4, pady=4)
                    self._thumb(card, path,
                                lambda m=members, mi=mi: open_full_image(self, m, mi)).pack()
                    split = yolo_split(path)
                    ttk.Label(card, text=f"{path.name} ({split})" if split else path.name,
                              style="Card.TLabel", wraplength=170).pack()
                    score = self.scores.get(str(path))
                    if score is not None:
                        ttk.Label(card, text=f"q={score:.2f}", style="Card.TLabel",
                                  foreground=COLOR_TEXT_MUTED).pack()
                    keep_var = tk.BooleanVar(value=mi in self.keep[ci])
                    ttk.Checkbutton(card, text="Keep this", variable=keep_var,
                                    command=lambda ci=ci, mi=mi, v=keep_var:
                                        self._set_keep(ci, mi, v)
                                    ).pack()

        def _set_keep(self, ci, mi, var):
            if var.get():
                self.keep[ci].add(mi)
            elif self.keep[ci] == {mi}:
                var.set(True)  # at least one image per cluster must be kept
                messagebox.showinfo(
                    "Keep at least one",
                    "Each cluster must keep at least one image.\n"
                    "Use \"Skip this cluster\" to keep them all.", parent=self)
                return
            else:
                self.keep[ci].discard(mi)
            self._update_summary()

        def _set_skip(self, ci, val):
            self.skip[ci] = val
            self._update_summary()

        def _set_all(self, include):
            self.skip = [not include] * len(self.clusters)
            for var in self.skip_vars.values():
                var.set(not include)
            self._update_summary()

        def _to_delete(self):
            return [p for ci, members in enumerate(self.clusters) if not self.skip[ci]
                    for mi, p in enumerate(members) if mi not in self.keep[ci]]

        def _update_summary(self):
            included = self.skip.count(False)
            self.summary_lbl.config(
                text=f"{included}/{len(self.clusters)} cluster(s) included · "
                     f"{len(self._to_delete())} file(s) to delete")

        def _auto_clean(self):
            self.destroy()
            self.on_auto_clean()

        def _confirm(self):
            to_delete = self._to_delete()
            if not to_delete:
                messagebox.showinfo("Nothing to delete", "All clusters skipped.", parent=self)
                return
            if not messagebox.askyesno("Confirm deletion",
                                       f"Remove {len(to_delete)} duplicate(s)?\n"
                                       f"For YOLO datasets, companion label files are removed as well.",
                                       parent=self):
                return
            deleted = self.on_delete(to_delete)
            messagebox.showinfo("Done", f"{deleted} file(s) removed.", parent=self)
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
            self._embedder_loading = False
            self._setup_style()
            self._setup_window()
            self._build_ui()
            self._poll()
            self._reload_embedder(self.config.embedder)

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
            self.root.minsize(900, 480)
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

            # Everything below the brand bar scrolls, so the activity log stays
            # reachable when the window is shorter than the content.
            self.page = ScrollableFrame(self.root, fill_height=True)
            self.page.pack(fill="both", expand=True)
            self.body = self.page.inner

            title_wrap = ttk.Frame(self.body, padding=(24, 18, 24, 4))
            title_wrap.pack(fill="x")
            ttk.Label(title_wrap, text="Dataset Cleaner", style="H1.TLabel").pack(anchor="w")
            ttk.Label(title_wrap,
                      text="Curate image datasets — detect blur, exposure issues, "
                           "low-information frames, and CLIP-based duplicates.",
                      style="Sub.TLabel").pack(anchor="w", pady=(2, 0))

            pathf = ttk.Frame(self.body, padding=(24, 12, 24, 4))
            pathf.pack(fill="x")
            ttk.Label(pathf, text="Dataset folder", style="H2.TLabel").pack(anchor="w")
            row = ttk.Frame(pathf)
            row.pack(fill="x", pady=(6, 0))
            self.path_entry = ttk.Entry(row, textvariable=self.dataset_path)
            self.path_entry.pack(side="left", fill="x", expand=True, ipady=4)
            ttk.Button(row, text="Browse…", command=self._browse).pack(side="left", padx=(8, 0))

            self._build_settings()

            actions = ttk.Frame(self.body, padding=(24, 10, 24, 4))
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
            self.clean_all_btn = ttk.Button(actions, text="Clean all problems",
                                            style="Danger.TButton",
                                            command=self._clean_all_problems,
                                            state="disabled")
            self.clean_all_btn.pack(side="left", padx=(8, 0))
            ttk.Button(actions, text="Open reports",
                       command=self._open_reports).pack(side="right")
            self.status_lbl = ttk.Label(actions, text="Loading embedder…",
                                        style="Sub.TLabel")
            self.status_lbl.pack(side="left", padx=14)

            self.progress = ttk.Progressbar(self.body, mode="determinate", maximum=100)
            self.progress.pack(fill="x", padx=24, pady=(4, 8))

            self.results_frame = ttk.Frame(self.body, padding=(20, 4, 20, 4))
            self.results_frame.pack(fill="x")
            self._render_result_cards(empty=True)

            logf = ttk.Frame(self.body, padding=(24, 8, 24, 16))
            logf.pack(fill="both", expand=True)
            ttk.Label(logf, text="Activity log", style="H2.TLabel").pack(anchor="w")
            log_box = tk.Frame(logf, bg=COLOR_BORDER, bd=0, highlightthickness=1,
                               highlightbackground=COLOR_BORDER)
            log_box.pack(fill="both", expand=True, pady=(6, 0))
            self.log = tk.Text(log_box, height=10, wrap="word",
                               bg=COLOR_SURFACE, fg=COLOR_TEXT,
                               relief="flat", borderwidth=0,
                               padx=10, pady=8,
                               font=("Consolas", 9))
            self.log.pack(side="left", fill="both", expand=True)
            sb = ttk.Scrollbar(log_box, orient="vertical", command=self.log.yview)
            sb.pack(side="right", fill="y")
            self.log.configure(yscrollcommand=sb.set, state="disabled")

        def _build_settings(self):
            wrap = ttk.Frame(self.body, padding=(24, 8, 24, 4))
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
            self.embedder_combo = ttk.Combobox(row3, textvariable=self.embedder_label_var,
                                               values=embedder_labels, width=42,
                                               state="readonly")
            self.embedder_combo.pack(side="left", padx=(0, 12))

            def _commit_embedder(*_):
                sel = self.embedder_label_var.get()
                idx = embedder_labels.index(sel) if sel in embedder_labels else 0
                key = embedder_keys[idx]
                if key == self.config.embedder and self.embedder is not None:
                    return
                previous = self.config.embedder if self.embedder is not None else None
                self.config.embedder = key
                self.config.save()
                self._reload_embedder(key, revert_to=previous)

            self.embedder_combo.bind("<<ComboboxSelected>>", _commit_embedder)

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
            any_problem = not empty and any(self._count(k) for k, _, _ in CATEGORY_META)
            self.clean_all_btn.state(["!disabled" if any_problem else "disabled"])

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
                    f"Keep one image in each of {len(clusters)} cluster(s) "
                    f"and remove {len(to_delete)} duplicate(s)?\n\n"
                    f"Copies in test/val are kept over train; otherwise the "
                    f"highest-quality one\n"
                    f"(sharpness + info + resolution − brightness extremes).\n"
                    f"Companion label files will be removed as well."):
                return
            self._delete_paths(to_delete)

        def _clean_all_problems(self):
            """Remove every flagged image plus all but one copy per duplicate cluster."""
            if not self.results or (self.job and self.job.is_alive()):
                return
            flagged: dict[Path, None] = {}  # insertion-ordered set
            lines = []
            for key, label, _ in CATEGORY_META:
                if key == "duplicates":
                    continue
                paths = [it[0] if isinstance(it, tuple) else it
                         for it in self.results.get(key, [])]
                if paths:
                    lines.append(f"  • {label}: {len(paths)}")
                flagged.update(dict.fromkeys(paths))
            n_dupes = 0
            for members in self.results.get("duplicate_clusters", []):
                # Keep the preferred copy among those not already flagged for quality.
                extra = [p for p in members if p not in flagged][1:]
                n_dupes += len(extra)
                flagged.update(dict.fromkeys(extra))
            if n_dupes:
                lines.append(f"  • Duplicates (all but one per cluster): {n_dupes}")
            if not flagged:
                messagebox.showinfo("Clean all problems", "Nothing to clean.")
                return
            target = ("the system trash" if self.trash_var.get() and HAS_SEND2TRASH
                      else "a quarantine folder (undoable)")
            labels_note = (" and their YOLO labels"
                           if self.results.get("dataset_format") == FORMAT_YOLO else "")
            if not messagebox.askyesno(
                    "Clean all problems",
                    f"Remove {len(flagged)} image(s){labels_note} to {target}?\n\n"
                    + "\n".join(lines) +
                    "\n\nImages flagged in several categories are counted once. In each "
                    "duplicate cluster, the first copy not already flagged is kept."):
                return
            self._delete_paths(list(flagged))

        def _open_reports(self):
            report_dir = APP_DIR / "reports"
            try:
                report_dir.mkdir(parents=True, exist_ok=True)
                if sys.platform.startswith("win"):
                    os.startfile(report_dir)  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(report_dir)])
                else:
                    subprocess.Popen(["xdg-open", str(report_dir)])
            except OSError as e:
                messagebox.showerror("Open reports", f"Could not open {report_dir}:\n{e}")

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
            with_label = self.results.get("dataset_format") == FORMAT_YOLO
            total = 0
            batch_moves: list[tuple[Path, Path]] = []
            for p in paths:
                count, moves = remove_yolo_pair(p, use_trash, quarantine, with_label)
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
                messagebox.showinfo("Saved", "API keys saved.", parent=win)
                win.destroy()
                # Retry a load that failed for lack of a key.
                if self.embedder is None and not self._embedder_loading:
                    self._reload_embedder(self.config.embedder)

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
            fmt, reason = detect_dataset_format(path)
            if fmt is None:
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
            self._set_running(True)
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
                self._set_running(False)
            elif evt.kind == "cancelled":
                self.status_lbl.config(text="Cancelled.")
                self._append_log("Analysis cancelled.")
                self.progress["value"] = 0
                self._set_running(False)
            elif evt.kind == "error":
                self.status_lbl.config(text="Error.")
                self._append_log(f"ERROR: {evt.message}")
                messagebox.showerror("Analysis error", evt.message)
                self._set_running(False)
            elif evt.kind == "embedder_ready":
                self._embedder_loading = False
                self.embedder = evt.payload
                spec = self.embedder.spec
                self.status_lbl.config(
                    text=f"Ready. {spec.label} loaded on {device.upper()}.")
                self._append_log(f"Embedder ready: {spec.label}")
                self._set_running(False)
            elif evt.kind == "embedder_error":
                self._embedder_loading = False
                self._set_running(False)
                self._append_log(f"Embedder load failed: {evt.message}")
                messagebox.showerror("Embedder error", evt.message)
                revert_to = evt.payload
                if revert_to and revert_to != self.config.embedder:
                    self.config.embedder = revert_to
                    self.config.save()
                    self.embedder_label_var.set(get_spec(revert_to).label)
                    self._append_log(f"Reverting to {get_spec(revert_to).label}.")
                    self._reload_embedder(revert_to)
                else:
                    self.status_lbl.config(
                        text="Embedder load failed — pick another one or set API keys.")

        def _set_running(self, running: bool):
            """Enable/disable controls for analysis runs and embedder loads."""
            busy = running or self._embedder_loading
            ready = not busy and self.embedder is not None
            self.analyze_btn.state(["!disabled" if ready else "disabled"])
            self.cancel_btn.state(["!disabled" if running else "disabled"])
            self.embedder_combo.state(["disabled" if busy else "!disabled"])

        def _reload_embedder(self, key: str, revert_to: Optional[str] = None):
            """Load embedder `key` on a background thread.

            The previous model is released first so two large models never
            share GPU memory. If loading fails and `revert_to` is given, the
            event handler switches back to that key.
            """
            self._embedder_loading = True
            self.embedder = None
            self.job = None  # a finished job still references the old model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            self._set_running(False)
            label = get_spec(key).label
            self.status_lbl.config(text=f"Loading {label} on {device.upper()}…")
            self._append_log(f"Loading embedder {label}…")

            def work():
                try:
                    emb = load_embedder(key, device)
                    self.event_q.put(Event(kind="embedder_ready", payload=emb))
                except Exception as e:
                    self.event_q.put(Event(kind="embedder_error", message=str(e),
                                           payload=revert_to))

            threading.Thread(target=work, daemon=True).start()

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
                   help="Keep one image per cluster (val/test copies first, "
                        "then best quality), remove others")
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

    fmt, reason = detect_dataset_format(args.dataset)
    if fmt is None:
        print(reason, file=sys.stderr)
        return 2
    print(f"{FORMAT_NAMES[fmt]} dataset ({reason}).")

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
                count, _ = remove_yolo_pair(path, use_trash, quarantine,
                                            with_label=(fmt == FORMAT_YOLO))
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
