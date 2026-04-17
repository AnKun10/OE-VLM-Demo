"""Qwen vLLM provider.

Wraps the OpenAI SDK client with Qwen-specific input transforms and
a one-retry policy on connection errors.
"""
from __future__ import annotations

import time

from openai import APIConnectionError, OpenAI

from ..base import VLMProvider
from . import config, transforms


class QwenVLLMProvider(VLMProvider):
    """Provider for Qwen-family VL models served via vLLM HTTP."""

    @classmethod
    def extra_kwargs_from_entry(cls, entry: dict) -> dict:
        """Extract min_pixels and max_pixels from a YAML model entry if present."""
        kwargs: dict = {}
        if "min_pixels" in entry:
            kwargs["min_pixels"] = entry["min_pixels"]
        if "max_pixels" in entry:
            kwargs["max_pixels"] = entry["max_pixels"]
        return kwargs

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_id: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> None:
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=config.REQUEST_TIMEOUT_S,
        )
        self._model_id = model_id
        self._base_url = base_url
        self._min_pixels = (
            min_pixels if min_pixels is not None else config.DEFAULT_MIN_PIXELS
        )
        self._max_pixels = (
            max_pixels if max_pixels is not None else config.DEFAULT_MAX_PIXELS
        )

    def generate(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        messages = transforms.strip_image_tokens(messages)
        messages = transforms.inject_pixel_bounds(
            messages, self._min_pixels, self._max_pixels
        )

        last_exc: APIConnectionError
        for attempt in range(config.MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.choices[0].message.content
                return (content or "").strip()
            except APIConnectionError as exc:
                last_exc = exc
                if attempt < config.MAX_RETRIES:
                    time.sleep(config.RETRY_BACKOFF_S)

        raise ConnectionError(
            f"Cannot connect to model '{self._model_id}' at {self._base_url}. "
            f"Is the vLLM server running? ({last_exc})"
        )
