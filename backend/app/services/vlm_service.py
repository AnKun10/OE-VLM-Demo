from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from app.config import settings

_model: Any | None = None
_processor: Any | None = None
_device: str | None = None


def load_vlm_model() -> None:
    global _model, _processor, _device
    if _model is not None:
        return

    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model_name = settings.vlm_model_name
    device_cfg = settings.vlm_device

    # Resolve device
    if device_cfg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_cfg

    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading VLM: {model_name} on {device} ({dtype}) ...")
    processor = AutoProcessor.from_pretrained(model_name)

    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
    except RuntimeError as exc:
        if device != "cuda" or "no kernel image" not in str(exc):
            raise
        print("CUDA not supported by current PyTorch build. Falling back to CPU.")
        device = "cpu"
        dtype = torch.float32
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()

    _model = model
    _processor = processor
    _device = device
    print(f"VLM loaded: {model_name} | device: {device}")


def unload_vlm_model() -> None:
    global _model, _processor, _device
    _model = None
    _processor = None
    _device = None


def is_loaded() -> bool:
    return _model is not None


@torch.no_grad()
def generate_response(
    prompt: str,
    image: Image.Image | None = None,
    max_new_tokens: int = 256,
) -> str:
    if _model is None or _processor is None:
        raise RuntimeError("VLM model is not loaded")

    if image is not None:
        conversation = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs = _processor(text=conversation, images=image, return_tensors="pt").to(
            _device, _model.dtype
        )
    else:
        conversation = f"USER: {prompt}\nASSISTANT:"
        inputs = _processor(text=conversation, return_tensors="pt").to(_device)

    output_ids = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Decode only the newly generated tokens (skip the input prompt tokens)
    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    response = _processor.decode(generated_ids, skip_special_tokens=True).strip()
    return response
