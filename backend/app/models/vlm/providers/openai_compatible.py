from __future__ import annotations

from typing import AsyncIterator

from openai import APIConnectionError, AsyncOpenAI, BadRequestError

from .base import VLMProvider


class OpenAICompatibleProvider(VLMProvider):
    """Provider for any OpenAI-compatible API (vLLM, OpenAI, etc.)."""

    def __init__(self, base_url: str, api_key: str, model_id: str) -> None:
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id
        self._base_url = base_url

    async def stream(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        try:
            result = await self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
        except BadRequestError as exc:
            if "max_completion_tokens" not in str(exc):
                raise
            # Older API (e.g. older vLLM) doesn't support max_completion_tokens
            result = await self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
        except APIConnectionError as exc:
            raise ConnectionError(
                f"Cannot connect to model '{self.model_id}' at {self._base_url}. "
                f"Is the model server running? ({exc})"
            ) from exc

        try:
            async for chunk in result:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except APIConnectionError as exc:
            raise ConnectionError(
                f"Connection lost while streaming from '{self.model_id}': {exc}"
            ) from exc
