"""API-backed image embedders.

Two flavors are supported:

  * Native image-embedding APIs (Jina, Voyage) — single HTTP call returns a
    cosine-comparable embedding vector per image.
  * Chat-VLM APIs (Qwen-VL, OpenAI gpt-4o-mini, Claude, Gemini) — caption each
    image at temperature 0 with a deterministic prompt, then embed the caption
    text with a tiny local SBERT model. Two identical images yield (near-)
    identical captions, so cosine on caption embeddings approximates visual
    duplicate detection.

API keys are resolved in this order:
  1. Environment variable (per family — see ``KEY_ENV_VARS``)
  2. ``~/.genow_cleaner/api_keys.json`` (managed via :meth:`KeyStore.set`)

All API embedders share a disk-backed cache at
``~/.genow_cleaner/embed_cache/<embedder_key>.jsonl``, keyed by SHA256 of the
normalized JPEG bytes — so repeated runs on the same image pay zero API cost.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger("genow_cleaner.api")

APP_DIR = Path.home() / ".genow_cleaner"
KEY_FILE = APP_DIR / "api_keys.json"
CACHE_DIR = APP_DIR / "embed_cache"

KEY_ENV_VARS = {
    "jina": "JINA_API_KEY",
    "voyage": "VOYAGE_API_KEY",
    "qwen-vl": "DASHSCOPE_API_KEY",
    "openai-vlm": "OPENAI_API_KEY",
    "anthropic-vlm": "ANTHROPIC_API_KEY",
    "gemini-vlm": "GEMINI_API_KEY",
}

CAPTION_PROMPT = (
    "Describe this image in 50–80 words. Focus exclusively on distinctive "
    "visual content: main subject, scene composition, key objects, dominant "
    "colors, perspective, and notable details. Be precise, factual, and "
    "consistent — two identical images must yield identical descriptions. "
    "Output the description only, no preamble."
)

MAX_IMAGE_SIDE = 1024
JPEG_QUALITY = 85


# ---------- API key storage ----------

class KeyStore:
    """Resolve and persist API keys. Env vars take precedence over the JSON file."""

    @staticmethod
    def get(family: str) -> str:
        env_var = KEY_ENV_VARS.get(family)
        if env_var:
            v = os.environ.get(env_var)
            if v:
                return v
        if KEY_FILE.exists():
            try:
                data = json.loads(KEY_FILE.read_text())
            except (OSError, json.JSONDecodeError):
                data = {}
            v = data.get(family)
            if v:
                return v
        raise RuntimeError(
            f"Missing API key for '{family}'. Set the {env_var} environment "
            f"variable or add it to {KEY_FILE}."
        )

    @staticmethod
    def has(family: str) -> bool:
        try:
            KeyStore.get(family)
            return True
        except RuntimeError:
            return False

    @staticmethod
    def set(family: str, key: str) -> None:
        KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {}
        if KEY_FILE.exists():
            try:
                data = json.loads(KEY_FILE.read_text())
            except (OSError, json.JSONDecodeError):
                data = {}
        if key:
            data[family] = key
        else:
            data.pop(family, None)
        KEY_FILE.write_text(json.dumps(data, indent=2))
        try:
            os.chmod(KEY_FILE, 0o600)
        except OSError:
            pass

    @staticmethod
    def listed_families() -> list[str]:
        return list(KEY_ENV_VARS.keys())


# ---------- Image normalization & hashing ----------

def _normalize_jpeg_bytes(img: Image.Image) -> bytes:
    """Resize to MAX_IMAGE_SIDE, encode as JPEG. Stable across runs."""
    work = img.copy()
    work.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buf = io.BytesIO()
    work.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def image_sha256(img: Image.Image) -> str:
    return hashlib.sha256(_normalize_jpeg_bytes(img)).hexdigest()


def _pil_to_b64_jpeg(img: Image.Image) -> str:
    return base64.b64encode(_normalize_jpeg_bytes(img)).decode("ascii")


# ---------- Disk cache ----------

class ImageEmbedCache:
    """JSON-lines disk cache: sha256 -> embedding (list of float)."""

    def __init__(self, embedder_key: str, cache_dir: Path = CACHE_DIR):
        safe = embedder_key.replace(":", "_").replace("/", "_")
        self.path = cache_dir / f"{safe}.jsonl"
        self._mem: dict[str, list[float]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    h = rec.get("h")
                    e = rec.get("e")
                    if isinstance(h, str) and isinstance(e, list):
                        self._mem[h] = e
        except OSError as exc:
            logger.warning("Embed cache read failed: %s", exc)

    def get(self, h: str) -> Optional[list[float]]:
        return self._mem.get(h)

    def put(self, h: str, emb: list[float]) -> None:
        if h in self._mem:
            return
        self._mem[h] = emb
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"h": h, "e": emb}) + "\n")
        except OSError as exc:
            logger.warning("Embed cache write failed: %s", exc)

    def __len__(self) -> int:
        return len(self._mem)


# ---------- Local text embedder (for caption-based VLMs) ----------

_TEXT_MODEL = None
_TEXT_TOK = None
_TEXT_DEVICE = "cpu"


def _get_text_embedder(device: str):
    global _TEXT_MODEL, _TEXT_TOK, _TEXT_DEVICE
    if _TEXT_MODEL is None:
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "Caption-based VLM embedders require 'transformers' "
                "(for the local SBERT text encoder). Install with: "
                "pip install transformers") from e
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        _TEXT_TOK = AutoTokenizer.from_pretrained(model_id)
        _TEXT_MODEL = AutoModel.from_pretrained(model_id).to(device).eval()
        _TEXT_DEVICE = device
    return _TEXT_MODEL, _TEXT_TOK, _TEXT_DEVICE


def _embed_texts(texts: list[str], device: str) -> np.ndarray:
    import torch
    model, tok, dev = _get_text_embedder(device)
    enc = tok(texts, padding=True, truncation=True, max_length=128,
              return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**enc)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    summed = (out.last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    feats = (summed / counts).cpu().numpy()
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    return (feats / norms).astype(np.float32)


# ---------- HTTP helpers ----------

def _require_requests() -> None:
    if not HAS_REQUESTS:
        raise RuntimeError("API embedders require 'requests'. "
                           "Install with: pip install requests")


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / norms).astype(np.float32)


# ---------- Native image embedding APIs ----------

def _jina_embed(api_key: str, model_id: str, pil_images: list) -> np.ndarray:
    _require_requests()
    payload = {
        "model": model_id,
        "input": [{"image": _pil_to_b64_jpeg(img)} for img in pil_images],
    }
    r = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json=payload, timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    embs = [d["embedding"] for d in data["data"]]
    return _l2_normalize(np.array(embs, dtype=np.float32))


def _voyage_embed(api_key: str, model_id: str, pil_images: list) -> np.ndarray:
    _require_requests()
    inputs = []
    for img in pil_images:
        b64 = _pil_to_b64_jpeg(img)
        inputs.append({"content": [{
            "type": "image_base64",
            "image_base64": f"data:image/jpeg;base64,{b64}",
        }]})
    r = requests.post(
        "https://api.voyageai.com/v1/multimodalembeddings",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={"model": model_id, "inputs": inputs, "input_type": "document"},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    embs = [d["embedding"] for d in data["data"]]
    return _l2_normalize(np.array(embs, dtype=np.float32))


# ---------- Chat-VLM captioning (one image per call) ----------

def _qwen_caption(api_key: str, model_id: str, pil_image: Image.Image) -> str:
    _require_requests()
    b64 = _pil_to_b64_jpeg(pil_image)
    r = requests.post(
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={
            "model": model_id,
            "temperature": 0,
            "max_tokens": 256,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": CAPTION_PROMPT},
                ],
            }],
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _openai_caption(api_key: str, model_id: str, pil_image: Image.Image) -> str:
    _require_requests()
    b64 = _pil_to_b64_jpeg(pil_image)
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={
            "model": model_id,
            "temperature": 0,
            "max_tokens": 256,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": CAPTION_PROMPT},
                ],
            }],
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _anthropic_caption(api_key: str, model_id: str, pil_image: Image.Image) -> str:
    _require_requests()
    b64 = _pil_to_b64_jpeg(pil_image)
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model_id,
            "max_tokens": 256,
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64}},
                    {"type": "text", "text": CAPTION_PROMPT},
                ],
            }],
        },
        timeout=180,
    )
    r.raise_for_status()
    data = r.json()
    # content is a list of blocks; concatenate text blocks
    return "".join(b.get("text", "") for b in data["content"]
                   if b.get("type") == "text").strip()


def _gemini_caption(api_key: str, model_id: str, pil_image: Image.Image) -> str:
    _require_requests()
    b64 = _pil_to_b64_jpeg(pil_image)
    url = (f"https://generativelanguage.googleapis.com/v1beta/"
           f"models/{model_id}:generateContent?key={api_key}")
    r = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                {"text": CAPTION_PROMPT},
            ]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 256},
        },
        timeout=180,
    )
    r.raise_for_status()
    data = r.json()
    parts = data["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts).strip()


# ---------- Embedder classes ----------

class _NativeAPIEmbedder:
    """Wraps a function that returns image embeddings directly."""

    def __init__(self, spec, device: str, family: str,
                 embed_fn: Callable[[str, str, list], np.ndarray]):
        self.spec = spec
        self.device = device
        self.family = family
        self.api_key = KeyStore.get(family)
        self.cache = ImageEmbedCache(spec.key)
        self._embed_fn = embed_fn

    def embed(self, pil_images: list) -> np.ndarray:
        out: list[Optional[list[float]]] = [None] * len(pil_images)
        pending_idx: list[int] = []
        pending_imgs: list[Image.Image] = []
        pending_h: list[str] = []
        for i, img in enumerate(pil_images):
            h = image_sha256(img)
            cached = self.cache.get(h)
            if cached is not None:
                out[i] = cached
            else:
                pending_idx.append(i)
                pending_imgs.append(img)
                pending_h.append(h)
        if pending_imgs:
            feats = self._embed_fn(self.api_key, self.spec.model_id, pending_imgs)
            for idx, h, vec in zip(pending_idx, pending_h, feats):
                lst = vec.tolist()
                self.cache.put(h, lst)
                out[idx] = lst
        return np.array(out, dtype=np.float32)


class _CaptionAPIEmbedder:
    """Caption each image via a chat VLM, then embed captions with local SBERT."""

    def __init__(self, spec, device: str, family: str,
                 caption_fn: Callable[[str, str, Image.Image], str]):
        self.spec = spec
        self.device = device
        self.family = family
        self.api_key = KeyStore.get(family)
        self.cache = ImageEmbedCache(spec.key)
        self._caption_fn = caption_fn

    def embed(self, pil_images: list) -> np.ndarray:
        out: list[Optional[list[float]]] = [None] * len(pil_images)
        pending_idx: list[int] = []
        pending_captions: list[str] = []
        pending_h: list[str] = []
        for i, img in enumerate(pil_images):
            h = image_sha256(img)
            cached = self.cache.get(h)
            if cached is not None:
                out[i] = cached
                continue
            try:
                caption = self._caption_fn(self.api_key, self.spec.model_id, img)
            except Exception as e:
                logger.warning("Caption failed for image %d (%s): %s", i, self.family, e)
                caption = ""
            pending_idx.append(i)
            pending_captions.append(caption or " ")
            pending_h.append(h)
        if pending_captions:
            feats = _embed_texts(pending_captions, self.device)
            for idx, h, vec in zip(pending_idx, pending_h, feats):
                lst = vec.tolist()
                self.cache.put(h, lst)
                out[idx] = lst
        return np.array(out, dtype=np.float32)


# ---------- Factory ----------

_NATIVE_FAMILIES = {
    "jina": _jina_embed,
    "voyage": _voyage_embed,
}

_CAPTION_FAMILIES = {
    "qwen-vl": _qwen_caption,
    "openai-vlm": _openai_caption,
    "anthropic-vlm": _anthropic_caption,
    "gemini-vlm": _gemini_caption,
}


def is_api_family(family: str) -> bool:
    return family in _NATIVE_FAMILIES or family in _CAPTION_FAMILIES


def make_api_embedder(spec, device: str):
    """Construct an API embedder. Raises RuntimeError if its API key is missing."""
    if spec.family in _NATIVE_FAMILIES:
        return _NativeAPIEmbedder(spec, device, spec.family,
                                   _NATIVE_FAMILIES[spec.family])
    if spec.family in _CAPTION_FAMILIES:
        return _CaptionAPIEmbedder(spec, device, spec.family,
                                    _CAPTION_FAMILIES[spec.family])
    raise ValueError(f"Not an API family: {spec.family}")
