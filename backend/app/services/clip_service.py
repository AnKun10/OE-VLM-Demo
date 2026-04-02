from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    MetaClip2TextModelWithProjection,
    MetaClip2VisionModelWithProjection,
)

from app.config import settings

MODEL_ID = settings.metaclip_model_id
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_text_model: Any | None = None
_vision_model: Any | None = None
_processor: Any | None = None
_tokenizer: Any | None = None
_vector_size: int | None = None
_runtime_device: str | None = None


def _move_inputs_to_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _load_models_on_device(device: str) -> tuple[Any, Any, Any, Any, int]:
    text_model = MetaClip2TextModelWithProjection.from_pretrained(MODEL_ID)
    vision_model = MetaClip2VisionModelWithProjection.from_pretrained(MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    text_model = text_model.to(device)
    vision_model = vision_model.to(device)
    text_model.eval()
    vision_model.eval()

    with torch.no_grad():
        dummy_inputs = tokenizer("test", padding=True, return_tensors="pt")
        dummy_inputs = _move_inputs_to_device(dummy_inputs, device)
        vector_size = int(text_model(**dummy_inputs).text_embeds.shape[-1])

    return text_model, vision_model, processor, tokenizer, vector_size


def load_clip_model() -> None:
    global _text_model, _vision_model, _processor, _tokenizer, _vector_size, _runtime_device
    if _text_model is not None and _vision_model is not None:
        return

    device = DEFAULT_DEVICE
    try:
        text_model, vision_model, processor, tokenizer, vector_size = _load_models_on_device(device)
    except RuntimeError as exc:
        if device != "cuda":
            raise
        print(f"CUDA load failed ({exc}). Falling back to CPU.")
        device = "cpu"
        text_model, vision_model, processor, tokenizer, vector_size = _load_models_on_device(device)

    _text_model = text_model
    _vision_model = vision_model
    _processor = processor
    _tokenizer = tokenizer
    _vector_size = vector_size
    _runtime_device = device

    print(f"Model: {MODEL_ID}")
    print(f"Vector size: {_vector_size}")
    print(f"Device: {device}")


def unload_clip_model() -> None:
    global _text_model, _vision_model, _processor, _tokenizer, _vector_size, _runtime_device
    _text_model = None
    _vision_model = None
    _processor = None
    _tokenizer = None
    _vector_size = None
    _runtime_device = None


def get_text_model() -> Any:
    load_clip_model()
    return _text_model


def get_vision_model() -> Any:
    load_clip_model()
    return _vision_model


def get_processor() -> Any:
    load_clip_model()
    return _processor


def get_tokenizer() -> Any:
    load_clip_model()
    return _tokenizer


def get_vector_size() -> int:
    load_clip_model()
    return int(_vector_size)


def get_runtime_device() -> str:
    load_clip_model()
    return str(_runtime_device)


@torch.no_grad()
def embed_image(pil_image: Image.Image) -> np.ndarray:
    processor = get_processor()
    model = get_vision_model()
    device = get_runtime_device()
    image_inputs = processor(images=pil_image, return_tensors="pt")
    image_inputs = _move_inputs_to_device(image_inputs, device)
    image_features = model(**image_inputs).image_embeds
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features[0].detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def embed_text(text: str) -> np.ndarray:
    tokenizer = get_tokenizer()
    model = get_text_model()
    device = get_runtime_device()
    text_inputs = tokenizer(text, padding=True, return_tensors="pt")
    text_inputs = _move_inputs_to_device(text_inputs, device)
    text_features = model(**text_inputs).text_embeds
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features[0].detach().cpu().numpy().astype(np.float32)
