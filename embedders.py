"""Pluggable image embedders for duplicate detection.

Each embedder loads lazily — heavy deps (torch, transformers, open_clip) are
imported only when that embedder is selected.

All embedders expose:
  - spec: EmbedderSpec
  - embed(pil_images: list[PIL.Image]) -> np.ndarray  (L2-normalized, float32)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class EmbedderSpec:
    key: str            # stable identifier stored in config (e.g. "clip:ViT-B/32")
    label: str          # human-readable label for UI
    family: str         # "clip" | "openclip" | "siglip" | "dinov2"
    model_id: str       # model identifier within the family
    suggested_threshold: float = 0.96


AVAILABLE_EMBEDDERS: list[EmbedderSpec] = [
    EmbedderSpec("clip:ViT-B/32", "CLIP ViT-B/32 (fast, default)",
                 "clip", "ViT-B/32", 0.96),
    EmbedderSpec("clip:ViT-B/16", "CLIP ViT-B/16",
                 "clip", "ViT-B/16", 0.95),
    EmbedderSpec("clip:ViT-L/14", "CLIP ViT-L/14 (strong)",
                 "clip", "ViT-L/14", 0.94),
    EmbedderSpec("openclip:ViT-H-14-laion2b",
                 "OpenCLIP ViT-H/14 laion2b  (best CLIP)",
                 "openclip", "ViT-H-14:laion2b_s32b_b79k", 0.93),
    EmbedderSpec("siglip:base-patch16-224",
                 "SigLIP base-224 (Google)",
                 "siglip", "google/siglip-base-patch16-224", 0.94),
    EmbedderSpec("siglip:large-patch16-256",
                 "SigLIP large-256",
                 "siglip", "google/siglip-large-patch16-256", 0.93),
    EmbedderSpec("dinov2:base", "DINOv2 base (strong visual dedup)",
                 "dinov2", "facebook/dinov2-base", 0.92),
    EmbedderSpec("dinov2:large", "DINOv2 large",
                 "dinov2", "facebook/dinov2-large", 0.91),
    # --- API-backed (require keys) ---
    EmbedderSpec("jina:clip-v2", "Jina CLIP v2  [API key]",
                 "jina", "jina-clip-v2", 0.90),
    EmbedderSpec("voyage:multimodal-3", "Voyage multimodal-3  [API key]",
                 "voyage", "voyage-multimodal-3", 0.90),
    EmbedderSpec("qwen-vl:plus", "Qwen-VL Plus (caption)  [API key]",
                 "qwen-vl", "qwen-vl-plus", 0.85),
    EmbedderSpec("qwen-vl:max", "Qwen-VL Max (caption)  [API key]",
                 "qwen-vl", "qwen-vl-max", 0.85),
    EmbedderSpec("openai-vlm:gpt-4o-mini", "OpenAI GPT-4o-mini (caption)  [API key]",
                 "openai-vlm", "gpt-4o-mini", 0.85),
    EmbedderSpec("anthropic-vlm:claude-sonnet-4-7",
                 "Claude Sonnet 4.7 (caption)  [API key]",
                 "anthropic-vlm", "claude-sonnet-4-7", 0.85),
    EmbedderSpec("gemini-vlm:2.0-flash", "Gemini 2.0 Flash (caption)  [API key]",
                 "gemini-vlm", "gemini-2.0-flash", 0.85),
]

DEFAULT_EMBEDDER_KEY = "clip:ViT-B/32"


def get_spec(key: str) -> EmbedderSpec:
    for s in AVAILABLE_EMBEDDERS:
        if s.key == key:
            return s
    raise ValueError(f"Unknown embedder key: {key}")


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / norms).astype(np.float32)


class _CLIPEmbedder:
    def __init__(self, spec: EmbedderSpec, device: str):
        import clip
        self.spec = spec
        self.device = device
        self.model, self.preprocess = clip.load(spec.model_id, device=device)
        self.model.eval()

    def embed(self, pil_images: list) -> np.ndarray:
        import torch
        tensors = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(tensors).cpu().numpy()
        return _l2_normalize(feats)


class _OpenCLIPEmbedder:
    def __init__(self, spec: EmbedderSpec, device: str):
        try:
            import open_clip
        except ImportError as e:
            raise RuntimeError(
                "OpenCLIP embedders require 'open_clip_torch'. "
                "Install with: pip install open_clip_torch") from e
        model_name, pretrained = spec.model_id.split(":", 1)
        self.spec = spec
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        self.model = self.model.to(device).eval()

    def embed(self, pil_images: list) -> np.ndarray:
        import torch
        tensors = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(tensors).cpu().numpy()
        return _l2_normalize(feats)


class _SigLIPEmbedder:
    def __init__(self, spec: EmbedderSpec, device: str):
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise RuntimeError(
                "SigLIP embedders require 'transformers'. "
                "Install with: pip install transformers") from e
        self.spec = spec
        self.device = device
        self.processor = AutoProcessor.from_pretrained(spec.model_id)
        self.model = AutoModel.from_pretrained(spec.model_id).to(device).eval()

    def embed(self, pil_images: list) -> np.ndarray:
        import torch
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs).cpu().numpy()
        return _l2_normalize(feats)


class _DINOv2Embedder:
    def __init__(self, spec: EmbedderSpec, device: str):
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as e:
            raise RuntimeError(
                "DINOv2 embedders require 'transformers'. "
                "Install with: pip install transformers") from e
        self.spec = spec
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(spec.model_id)
        self.model = AutoModel.from_pretrained(spec.model_id).to(device).eval()

    def embed(self, pil_images: list) -> np.ndarray:
        import torch
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
        pooler = getattr(out, "pooler_output", None)
        if pooler is not None:
            feats = pooler.cpu().numpy()
        else:
            feats = out.last_hidden_state[:, 0, :].cpu().numpy()
        return _l2_normalize(feats)


def load_embedder(key: str, device: str):
    """Load an embedder by key. Raises RuntimeError if its optional deps or
    API key are missing."""
    spec = get_spec(key)
    if spec.family == "clip":
        return _CLIPEmbedder(spec, device)
    if spec.family == "openclip":
        return _OpenCLIPEmbedder(spec, device)
    if spec.family == "siglip":
        return _SigLIPEmbedder(spec, device)
    if spec.family == "dinov2":
        return _DINOv2Embedder(spec, device)
    from api_embedders import is_api_family, make_api_embedder
    if is_api_family(spec.family):
        return make_api_embedder(spec, device)
    raise ValueError(f"Unsupported family: {spec.family}")
