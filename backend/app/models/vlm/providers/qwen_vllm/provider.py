"""Qwen vLLM provider (async).

Wraps the AsyncOpenAI SDK client with Qwen-specific input transforms and
a one-retry policy on connection errors that fail BEFORE the first chunk
is yielded. Errors that occur after the first chunk propagate as
ConnectionError without retrying — the partial output is preserved by
the SSE handler upstream.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from openai import APIConnectionError, AsyncOpenAI

from ..base import VLMProvider
from . import config, transforms


class QwenVLLMProvider(VLMProvider):
    """Provider for Qwen-family VL models served via vLLM HTTP."""

    @classmethod
    def extra_kwargs_from_entry(cls, entry: dict) -> dict:
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
        self._client = AsyncOpenAI(
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

    async def stream(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        prepared = transforms.strip_image_tokens(messages)
        prepared = transforms.inject_pixel_bounds(
            prepared, self._min_pixels, self._max_pixels
        )

        result = None
        last_exc: APIConnectionError | None = None
        for attempt in range(config.MAX_RETRIES + 1):
            try:
                result = await self._client.chat.completions.create(
                    model=self._model_id,
                    messages=prepared,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )
                break
            except APIConnectionError as exc:
                last_exc = exc
                if attempt < config.MAX_RETRIES:
                    await asyncio.sleep(config.RETRY_BACKOFF_S)
                    continue

        if result is None:
            assert last_exc is not None
            raise ConnectionError(
                f"Cannot connect to model '{self._model_id}' at {self._base_url}. "
                f"Is the vLLM server running? ({last_exc})"
            )

        try:
            async for chunk in result:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except APIConnectionError as exc:
            raise ConnectionError(
                f"Connection lost while streaming from '{self._model_id}': {exc}"
            ) from exc
