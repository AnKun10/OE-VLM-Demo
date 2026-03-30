from __future__ import annotations

from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image

from app.config import settings

MODEL_NAME = settings.clip_model_name
PRETRAINED = settings.clip_pretrained
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model: Any | None = None
_preprocess: Any | None = None
_tokenizer: Any | None = None
_vector_size: int | None = None
_runtime_device: str | None = None


def load_clip_model() -> None:
    global _model, _preprocess, _tokenizer, _vector_size, _runtime_device
    if _model is not None:
        return

    device = DEFAULT_DEVICE
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    try:
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            dummy = tokenizer(["test"]).to(device)
            text_features = model.encode_text(dummy)
            vector_size = int(text_features.shape[-1])
    except RuntimeError as exc:
        if device != "cuda":
            raise
        if "no kernel image is available" not in str(exc):
            raise

        print("CUDA is available but not supported by the current PyTorch build. Falling back to CPU.")
        device = "cpu"
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            dummy = tokenizer(["test"]).to(device)
            text_features = model.encode_text(dummy)
            vector_size = int(text_features.shape[-1])

    _model = model
    _preprocess = preprocess
    _tokenizer = tokenizer
    _vector_size = vector_size
    _runtime_device = device

    print(f"Model: {MODEL_NAME} | pretrained: {PRETRAINED}")
    print(f"Vector size: {_vector_size}")
    print(f"Device: {device}")


def unload_clip_model() -> None:
    global _model, _preprocess, _tokenizer, _vector_size, _runtime_device
    _model = None
    _preprocess = None
    _tokenizer = None
    _vector_size = None
    _runtime_device = None


def get_model() -> Any:
    load_clip_model()
    return _model


def get_preprocess() -> Any:
    load_clip_model()
    return _preprocess


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
    preprocess = get_preprocess()
    model = get_model()
    device = get_runtime_device()
    image_input = preprocess(pil_image).unsqueeze(0).to(device)
    image_features = model.encode_image(image_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features[0].detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def embed_text(text: str) -> np.ndarray:
    tokenizer = get_tokenizer()
    model = get_model()
    device = get_runtime_device()
    text_tokens = tokenizer([text]).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features[0].detach().cpu().numpy().astype(np.float32)
