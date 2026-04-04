from __future__ import annotations

from typing import Any

from PIL import Image

from app.config import settings

_llm: Any | None = None
_sampling_params: Any | None = None


def load_vlm_model() -> None:
    global _llm, _sampling_params
    if _llm is not None:
        return

    from vllm import LLM, SamplingParams

    model_name = settings.vlm_model_name

    print(f"Loading VLM via vLLM: {model_name} ...")
    _llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
    )
    _sampling_params = SamplingParams(max_tokens=256, temperature=0)
    print(f"VLM loaded via vLLM: {model_name}")


def unload_vlm_model() -> None:
    global _llm, _sampling_params
    _llm = None
    _sampling_params = None


def is_loaded() -> bool:
    return _llm is not None


def generate_response(
    prompt: str,
    image: Image.Image | None = None,
    max_new_tokens: int = 256,
) -> str:
    if _llm is None or _sampling_params is None:
        raise RuntimeError("VLM model is not loaded")

    from vllm import SamplingParams

    params = SamplingParams(max_tokens=max_new_tokens, temperature=0)

    if image is not None:
        conversation = f"USER: <image>\n{prompt}\nASSISTANT:"
        outputs = _llm.generate(
            {
                "prompt": conversation,
                "multi_modal_data": {"image": image},
            },
            sampling_params=params,
        )
    else:
        conversation = f"USER: {prompt}\nASSISTANT:"
        outputs = _llm.generate(conversation, sampling_params=params)

    return outputs[0].outputs[0].text.strip()
