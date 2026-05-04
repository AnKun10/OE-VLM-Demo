from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

from app.config import settings

MODEL_ID = settings.fgclip_model_id
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_WALK_TYPE_MAX_LENGTH: dict[str, int] = {"short": 64, "long": 196}
_SHORT_TOKEN_THRESHOLD = _WALK_TYPE_MAX_LENGTH["short"]

_model: Any | None = None
_tokenizer: Any | None = None
_image_processor: Any | None = None
_fusion_text_vec: np.ndarray | None = None
_vector_size: int | None = None
_runtime_device: str | None = None


def _move_inputs_to_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(x))
    if denom <= 1e-12:
        return x
    return (x / denom).astype(np.float32)


def _detect_method_2(pil_rgba: Image.Image) -> bool:
    rgba = np.array(pil_rgba.convert("RGBA"))
    alpha = rgba[:, :, 3].astype(np.float32)
    h, w = alpha.shape
    fg_mask = alpha > 5
    fg_count = int(fg_mask.sum())
    if fg_count < 20:
        return False
    transparent_ratio = float((~fg_mask).sum()) / max(h * w, 1)
    return transparent_ratio > 0.05


def _apply_method_2(pil_rgba: Image.Image, bg: tuple[int, int, int] = (127, 127, 127)) -> Image.Image:
    rgba = np.array(pil_rgba.convert("RGBA")).astype(np.float32)
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3:4] / 255.0
    bg_arr = np.full_like(rgb, bg, dtype=np.float32)
    out = rgb * alpha + bg_arr * (1.0 - alpha)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _determine_max_patches(image: Image.Image) -> int:
    w, h = image.size
    max_val = (w // 16) * (h // 16)
    if max_val > 784:
        return 1024
    if max_val > 576:
        return 784
    if max_val > 256:
        return 576
    if max_val > 128:
        return 256
    return 128


def _repair_text_embeddings(model: Any, device: str) -> None:
    # FG-CLIP 2's remote code registers `position_ids` as a non-persistent buffer twice
    # via `torch.arange(...).expand((1, -1))`; under torch >= 2.11 the second registration's
    # storage is invalidated, leaving garbage indices that crash position_embedding lookup.
    # `mask1`/`mask2` are plain attributes that come back as meta tensors after from_pretrained.
    # Rebuild them with valid values so both walk_type="short" and "long" paths work.
    text_emb = model.text_model.embeddings
    tcfg = model.config.text_config
    longtext_len = int(tcfg.longtext_len)
    keep_len = int(tcfg.keep_len)

    text_emb.register_buffer(
        "position_ids",
        torch.arange(longtext_len, device=device).unsqueeze(0).contiguous(),
        persistent=False,
    )
    mask1 = torch.zeros([longtext_len, 1])
    mask1[:keep_len, :] = 1
    mask2 = torch.zeros([longtext_len, 1])
    mask2[keep_len:, :] = 1
    text_emb.mask1 = mask1
    text_emb.mask2 = mask2


def _load_on_device(device: str):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    _repair_text_embeddings(model, device)

    with torch.no_grad():
        probe_walk_type = "short"
        probe_max_length = _WALK_TYPE_MAX_LENGTH[probe_walk_type]
        dummy_tokens = tokenizer(
            ["probe"],
            padding="max_length",
            truncation=True,
            max_length=probe_max_length,
            return_tensors="pt",
        )
        dummy_tokens = _move_inputs_to_device(dummy_tokens, device)
        feat = model.get_text_features(**dummy_tokens, walk_type=probe_walk_type)
        vector_size = int(feat.shape[-1])
    return model, tokenizer, image_processor, vector_size


def load_clip_model() -> None:
    global _model, _tokenizer, _image_processor, _fusion_text_vec, _vector_size, _runtime_device
    if _model is not None:
        return

    device = DEFAULT_DEVICE
    try:
        model, tokenizer, image_processor, vector_size = _load_on_device(device)
    except RuntimeError as exc:
        if device != "cuda":
            raise
        print(f"CUDA load failed ({exc}). Falling back to CPU.")
        device = "cpu"
        model, tokenizer, image_processor, vector_size = _load_on_device(device)

    _model = model
    _tokenizer = tokenizer
    _image_processor = image_processor
    _vector_size = vector_size
    _runtime_device = device

    _fusion_text_vec = embed_text(settings.fusion_text)

    print(f"Model: {MODEL_ID}")
    print(f"Vector size: {_vector_size}")
    print(f"Device: {device}")


def unload_clip_model() -> None:
    global _model, _tokenizer, _image_processor, _fusion_text_vec, _vector_size, _runtime_device
    _model = None
    _tokenizer = None
    _image_processor = None
    _fusion_text_vec = None
    _vector_size = None
    _runtime_device = None


def get_vector_size() -> int:
    load_clip_model()
    return int(_vector_size)


def get_runtime_device() -> str:
    load_clip_model()
    return str(_runtime_device)


@torch.no_grad()
def embed_text(text: str) -> np.ndarray:
    load_clip_model()
    device = _runtime_device
    cleaned = text.lower().strip()

    probe = _tokenizer([cleaned], padding=False, truncation=False, return_tensors="pt")
    real_len = int(probe["input_ids"].shape[-1])
    walk_type = "short" if real_len <= _SHORT_TOKEN_THRESHOLD else "long"
    max_length = _WALK_TYPE_MAX_LENGTH[walk_type]

    tokens = _tokenizer(
        [cleaned],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokens = _move_inputs_to_device(tokens, device)
    feat = _model.get_text_features(**tokens, walk_type=walk_type)
    feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    return feat[0].detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def embed_image(pil_rgba: Image.Image) -> np.ndarray:
    load_clip_model()
    device = _runtime_device
    if _detect_method_2(pil_rgba):
        pil_rgb = _apply_method_2(pil_rgba)
    else:
        pil_rgb = pil_rgba.convert("RGB")
    image_input = _image_processor(
        images=pil_rgb,
        max_num_patches=_determine_max_patches(pil_rgb),
        return_tensors="pt",
    )
    image_input = _move_inputs_to_device(image_input, device)
    feat = _model.get_image_features(**image_input)
    feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    return feat[0].detach().cpu().numpy().astype(np.float32)


def early_fusion_embed(pil_rgba: Image.Image) -> np.ndarray:
    load_clip_model()
    img_vec = embed_image(pil_rgba)
    fused = settings.fusion_weight_image * img_vec + settings.fusion_weight_text * _fusion_text_vec
    return _l2_normalize(fused)
