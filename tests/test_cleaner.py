"""Tests for analysis primitives — no GUI, no embedder, no torch required."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genowCleaner import (  # noqa: E402
    Config,
    analyze_image_quality,
    cluster_by_similarity,
    FORMAT_IMAGES,
    FORMAT_YOLO,
    cluster_splits,
    collect_images,
    detect_dataset_format,
    keep_priority,
    quality_score,
    remove_yolo_pair,
    validate_yolo_dataset,
    write_report,
    yolo_label_path,
    yolo_split,
)


def _make_image(path: Path, color=(128, 128, 128), size=(64, 64), noise=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    if noise:
        rng = np.random.default_rng(0)
        arr = (arr.astype(np.int16) + rng.integers(-40, 40, arr.shape)).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _make_yolo_dataset(root: Path, splits=("train", "val")) -> None:
    """Build a minimal YOLO directory layout under root with one image per split."""
    (root / "data.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: object\n"
        "nc: 1\n"
    )
    for sp in splits:
        img = root / "images" / sp / f"{sp}_a.png"
        lbl = root / "labels" / sp / f"{sp}_a.txt"
        _make_image(img)
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("0 0.5 0.5 0.2 0.2\n")


def test_quality_score_handles_empty():
    assert quality_score({}) == float("-inf")


def test_quality_score_orders_sharp_above_blurry():
    sharp = {"blur": 500.0, "brightness": 128.0, "std": 50.0, "pixels": 10000}
    blurry = {"blur": 5.0, "brightness": 128.0, "std": 50.0, "pixels": 10000}
    assert quality_score(sharp) > quality_score(blurry)


def test_analyze_image_quality_reads_png(tmp_path):
    p = tmp_path / "test.png"
    _make_image(p, color=(200, 200, 200), noise=True)
    m = analyze_image_quality(str(p))
    assert m is not None
    assert m["pixels"] == 64 * 64
    assert 0 <= m["brightness"] <= 255


def test_analyze_image_quality_returns_none_for_missing():
    assert analyze_image_quality("/no/such/file.jpg") is None


def test_collect_images_skips_labels_dir_and_quarantine(tmp_path):
    _make_yolo_dataset(tmp_path)
    # Add a stray .png inside labels/ — must NOT be collected as an image
    stray = tmp_path / "labels" / "train" / "stray.png"
    _make_image(stray)
    nested = tmp_path / "_quarantine_old"
    _make_image(nested / "skip.png")

    found = collect_images(tmp_path)
    names = {p.name for p in found}
    assert "train_a.png" in names and "val_a.png" in names
    assert "stray.png" not in names
    assert "skip.png" not in names


def test_cluster_by_similarity_groups_identical():
    rng = np.random.default_rng(0)
    v = rng.normal(size=512).astype(np.float32)
    v /= np.linalg.norm(v)
    other = rng.normal(size=512).astype(np.float32)
    other /= np.linalg.norm(other)
    embeddings = np.stack([v, v, v, other])
    clusters = cluster_by_similarity(embeddings, threshold=0.95)
    assert len(clusters) == 1
    assert sorted(clusters[0]) == [0, 1, 2]


def _unit(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    return np.array([np.cos(a), np.sin(a)], dtype=np.float32)


def test_cluster_by_similarity_does_not_chain():
    # cos(20°)≈0.94 > 0.9 for neighbors, cos(40°)≈0.77 < 0.9 for A–C.
    embeddings = np.stack([_unit(0), _unit(20), _unit(40)])
    clusters = cluster_by_similarity(embeddings, threshold=0.9)
    assert clusters == [[0, 1]]


def test_cluster_by_similarity_leader_follows_order():
    embeddings = np.stack([_unit(0), _unit(20), _unit(40)])
    clusters = cluster_by_similarity(embeddings, threshold=0.9, order=[1, 0, 2])
    # B is visited first and is similar to both A and C.
    assert len(clusters) == 1
    assert clusters[0][0] == 1
    assert sorted(clusters[0]) == [0, 1, 2]


def test_cluster_by_similarity_empty():
    embeddings = np.zeros((1, 512), dtype=np.float32)
    assert cluster_by_similarity(embeddings, 0.9) == []


# ----- YOLO format support -----

def test_validate_yolo_dataset_accepts_data_yaml(tmp_path):
    _make_yolo_dataset(tmp_path)
    ok, reason = validate_yolo_dataset(tmp_path)
    assert ok, reason
    assert "data.yaml" in reason or "data file" in reason


def test_validate_yolo_dataset_accepts_images_labels_layout(tmp_path):
    _make_image(tmp_path / "images" / "train" / "a.png")
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.1 0.1")
    ok, reason = validate_yolo_dataset(tmp_path)
    assert ok, reason


def test_validate_yolo_dataset_rejects_plain_folder(tmp_path):
    _make_image(tmp_path / "a.png")
    _make_image(tmp_path / "b.png")
    ok, reason = validate_yolo_dataset(tmp_path)
    assert not ok
    assert "YOLO" in reason


def test_validate_yolo_dataset_rejects_yaml_without_yolo_keys(tmp_path):
    (tmp_path / "data.yaml").write_text("foo: bar\n")
    _make_image(tmp_path / "a.png")
    ok, _ = validate_yolo_dataset(tmp_path)
    assert not ok


def test_detect_dataset_format_yolo(tmp_path):
    _make_yolo_dataset(tmp_path)
    fmt, _ = detect_dataset_format(tmp_path)
    assert fmt == FORMAT_YOLO


def test_detect_dataset_format_plain_folder(tmp_path):
    _make_image(tmp_path / "a.png")
    fmt, _ = detect_dataset_format(tmp_path)
    assert fmt == FORMAT_IMAGES


def test_detect_dataset_format_rejects_missing_dir(tmp_path):
    fmt, reason = detect_dataset_format(tmp_path / "nope")
    assert fmt is None
    assert "Not a directory" in reason


def test_collect_images_keeps_labels_dir_for_plain_folder(tmp_path):
    _make_image(tmp_path / "labels" / "a.png")
    assert collect_images(tmp_path) == []
    assert collect_images(tmp_path, skip_labels=False) == [tmp_path / "labels" / "a.png"]


def test_remove_without_label_keeps_sibling_txt(tmp_path):
    img = tmp_path / "a.png"
    txt = tmp_path / "a.txt"
    _make_image(img)
    txt.write_text("caption")
    count, _ = remove_yolo_pair(img, use_trash=False,
                                quarantine_dir=tmp_path / "q", with_label=False)
    assert count == 1
    assert not img.exists()
    assert txt.exists()


def test_yolo_label_path_swaps_images_to_labels():
    img = Path("/data/proj/images/train/sub/x.jpg")
    assert yolo_label_path(img) == Path("/data/proj/labels/train/sub/x.txt")


def test_yolo_label_path_uses_rightmost_images_segment():
    img = Path("/images/dataset/images/val/x.png")
    assert yolo_label_path(img) == Path("/images/dataset/labels/val/x.txt")


def test_yolo_label_path_falls_back_to_sibling_txt():
    img = Path("/some/dir/x.jpg")
    assert yolo_label_path(img) == Path("/some/dir/x.txt")


def test_yolo_split_layouts():
    assert yolo_split(Path("/d/images/train/x.jpg")) == "train"
    assert yolo_split(Path("/d/valid/images/x.jpg")) == "val"
    assert yolo_split(Path("/d/images/test/sub/x.jpg")) == "test"
    assert yolo_split(Path("/d/images/x.jpg")) is None


def test_keep_priority_prefers_eval_split_over_quality():
    train = Path("/d/images/train/a.jpg")
    val = Path("/d/images/val/a.jpg")
    quality = {
        train: {"blur": 900.0, "brightness": 128.0, "std": 60.0, "pixels": 10**6},
        val: {"blur": 5.0, "brightness": 128.0, "std": 10.0, "pixels": 10**4},
    }
    assert sorted([train, val], key=lambda p: keep_priority(p, quality))[0] == val


def test_keep_priority_uses_quality_within_split():
    sharp = Path("/d/images/train/a.jpg")
    blurry = Path("/d/images/train/b.jpg")
    quality = {
        sharp: {"blur": 900.0, "brightness": 128.0, "std": 60.0, "pixels": 10**4},
        blurry: {"blur": 5.0, "brightness": 128.0, "std": 60.0, "pixels": 10**4},
    }
    assert sorted([blurry, sharp], key=lambda p: keep_priority(p, quality))[0] == sharp


def test_cluster_splits():
    members = [Path("/d/images/val/a.jpg"), Path("/d/images/train/a.jpg"),
               Path("/d/images/train/b.jpg")]
    assert cluster_splits(members) == ["train", "val"]


def test_remove_yolo_pair_quarantines_image_and_label(tmp_path):
    _make_yolo_dataset(tmp_path)
    img = tmp_path / "images" / "train" / "train_a.png"
    lbl = tmp_path / "labels" / "train" / "train_a.txt"
    assert img.exists() and lbl.exists()
    quarantine = tmp_path / "q"
    count, moves = remove_yolo_pair(img, use_trash=False, quarantine_dir=quarantine)
    assert count == 2
    assert not img.exists() and not lbl.exists()
    moved_names = {m[1].name for m in moves}
    assert moved_names == {"train_a.png", "train_a.txt"}


def test_remove_yolo_pair_handles_missing_label(tmp_path):
    """An image with no label still gets removed; we don't require labels."""
    img = tmp_path / "images" / "train" / "a.png"
    _make_image(img)
    quarantine = tmp_path / "q"
    count, _ = remove_yolo_pair(img, use_trash=False, quarantine_dir=quarantine)
    assert count == 1


def test_remove_yolo_pair_handles_collisions(tmp_path):
    img1 = tmp_path / "images" / "train" / "a.png"
    img2 = tmp_path / "images" / "val" / "a.png"
    _make_image(img1)
    _make_image(img2)
    q = tmp_path / "q"
    remove_yolo_pair(img1, use_trash=False, quarantine_dir=q)
    remove_yolo_pair(img2, use_trash=False, quarantine_dir=q)
    files = sorted(p.name for p in q.iterdir())
    assert files == ["a.png", "a_1.png"]


def test_write_report_structure(tmp_path):
    cfg = Config()
    img = tmp_path / "images" / "train" / "img.png"
    _make_image(img)
    results = {
        "blurry": [(img, 5.0)],
        "dark": [],
        "bright": [],
        "low_info": [],
        "name_filtered": [],
        "unreadable": [],
        "duplicate_clusters": [[img, img]],
        "quality_scores": {str(img): 1.23},
        "total_scanned": 1,
    }
    out = tmp_path / "report.json"
    write_report(out, tmp_path, results, cfg)
    import json
    data = json.loads(out.read_text())
    assert data["schema"] == 2
    assert data["dataset_format"] == "YOLOv11/v12"

    results["dataset_format"] = FORMAT_IMAGES
    write_report(out, tmp_path, results, cfg)
    assert json.loads(out.read_text())["dataset_format"] == "image folder"
    assert data["categories"]["blurry"][0]["score"] == 5.0
    assert data["duplicate_clusters"][0][0]["quality_score"] == 1.23
    assert data["duplicate_clusters"][0][0]["split"] == "train"


def test_config_roundtrip(tmp_path, monkeypatch):
    import genowCleaner
    monkeypatch.setattr(genowCleaner, "SETTINGS_FILE", tmp_path / "s.json")
    c = Config(blur_threshold=42.5, embedder="clip:ViT-L/14")
    c.save()
    loaded = Config.load()
    assert loaded.blur_threshold == 42.5
    assert loaded.embedder == "clip:ViT-L/14"


# ----- Embedder registry (no model load) -----

def test_embedder_registry_has_clip_default():
    from embedders import AVAILABLE_EMBEDDERS, DEFAULT_EMBEDDER_KEY, get_spec
    keys = {s.key for s in AVAILABLE_EMBEDDERS}
    assert DEFAULT_EMBEDDER_KEY in keys
    spec = get_spec(DEFAULT_EMBEDDER_KEY)
    assert spec.family == "clip"


def test_embedder_registry_covers_expected_families():
    from embedders import AVAILABLE_EMBEDDERS
    families = {s.family for s in AVAILABLE_EMBEDDERS}
    assert families == {
        "clip", "openclip", "siglip", "dinov2",
        "jina", "voyage", "qwen-vl", "openai-vlm", "anthropic-vlm", "gemini-vlm",
    }


def test_embedder_get_spec_raises_on_unknown():
    from embedders import get_spec
    with pytest.raises(ValueError):
        get_spec("nope:nada")


def test_qwen_vl_registered():
    from embedders import AVAILABLE_EMBEDDERS
    keys = {s.key for s in AVAILABLE_EMBEDDERS}
    assert "qwen-vl:plus" in keys and "qwen-vl:max" in keys


# ----- API embedder infrastructure (no network) -----

def test_is_api_family_classification():
    from api_embedders import is_api_family
    assert is_api_family("jina")
    assert is_api_family("qwen-vl")
    assert is_api_family("gemini-vlm")
    assert not is_api_family("clip")
    assert not is_api_family("dinov2")


def test_keystore_env_takes_precedence(tmp_path, monkeypatch):
    import api_embedders
    monkeypatch.setattr(api_embedders, "KEY_FILE", tmp_path / "keys.json")
    monkeypatch.setenv("JINA_API_KEY", "env-key-123")
    api_embedders.KeyStore.set("jina", "file-key-999")
    assert api_embedders.KeyStore.get("jina") == "env-key-123"


def test_keystore_falls_back_to_file(tmp_path, monkeypatch):
    import api_embedders
    monkeypatch.setattr(api_embedders, "KEY_FILE", tmp_path / "keys.json")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    assert not api_embedders.KeyStore.has("voyage")
    api_embedders.KeyStore.set("voyage", "file-secret")
    assert api_embedders.KeyStore.get("voyage") == "file-secret"
    assert api_embedders.KeyStore.has("voyage")


def test_keystore_missing_raises(tmp_path, monkeypatch):
    import api_embedders
    monkeypatch.setattr(api_embedders, "KEY_FILE", tmp_path / "nope.json")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        api_embedders.KeyStore.get("openai-vlm")


def test_image_sha256_stable_and_content_sensitive(tmp_path):
    from api_embedders import image_sha256
    a = Image.new("RGB", (64, 64), (10, 20, 30))
    b = Image.new("RGB", (64, 64), (10, 20, 30))
    c = Image.new("RGB", (64, 64), (200, 50, 50))
    assert image_sha256(a) == image_sha256(b)
    assert image_sha256(a) != image_sha256(c)


def test_image_embed_cache_roundtrip_and_persistence(tmp_path):
    from api_embedders import ImageEmbedCache
    c1 = ImageEmbedCache("jina:clip-v2", cache_dir=tmp_path)
    assert c1.get("abc") is None
    c1.put("abc", [0.1, 0.2, 0.3])
    assert c1.get("abc") == [0.1, 0.2, 0.3]
    # New instance reads the same file
    c2 = ImageEmbedCache("jina:clip-v2", cache_dir=tmp_path)
    assert c2.get("abc") == [0.1, 0.2, 0.3]
    assert len(c2) == 1


# ----- API retry (no network) -----

class _FakeResponse:
    def __init__(self, status: int, headers=None):
        self.status_code = status
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _patch_post(monkeypatch, responses):
    import api_embedders
    calls = []
    sleeps = []

    def fake_post(url, **kwargs):
        calls.append(url)
        r = responses[len(calls) - 1]
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(api_embedders.requests, "post", fake_post)
    monkeypatch.setattr(api_embedders.time, "sleep", sleeps.append)
    return calls, sleeps


def test_post_with_retry_recovers_from_429_and_honors_retry_after(monkeypatch):
    pytest.importorskip("requests")
    from api_embedders import _post_with_retry
    calls, sleeps = _patch_post(monkeypatch, [
        _FakeResponse(429, {"Retry-After": "7"}), _FakeResponse(503), _FakeResponse(200)])
    r = _post_with_retry("https://x/y")
    assert r.status_code == 200
    assert len(calls) == 3
    assert sleeps == [7.0, 4.0]


def test_post_with_retry_retries_network_errors(monkeypatch):
    requests = pytest.importorskip("requests")
    from api_embedders import _post_with_retry
    calls, _ = _patch_post(monkeypatch, [
        requests.ConnectionError("boom"), _FakeResponse(200)])
    assert _post_with_retry("https://x/y").status_code == 200
    assert len(calls) == 2


def test_post_with_retry_does_not_retry_client_errors(monkeypatch):
    pytest.importorskip("requests")
    from api_embedders import _post_with_retry
    calls, sleeps = _patch_post(monkeypatch, [_FakeResponse(401)])
    with pytest.raises(RuntimeError):
        _post_with_retry("https://x/y")
    assert len(calls) == 1 and sleeps == []


def test_post_with_retry_gives_up(monkeypatch):
    pytest.importorskip("requests")
    import api_embedders
    calls, _ = _patch_post(monkeypatch, [_FakeResponse(500)] * api_embedders.MAX_ATTEMPTS)
    with pytest.raises(RuntimeError):
        api_embedders._post_with_retry("https://x/y")
    assert len(calls) == api_embedders.MAX_ATTEMPTS
