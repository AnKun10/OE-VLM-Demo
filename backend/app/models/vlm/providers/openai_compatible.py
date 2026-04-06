# backend/app/models/vlm/providers/openai_compatible.py
from __future__ import annotations

from openai import APIConnectionError, BadRequestError, OpenAI

from .base import VLMProvider


class OpenAICompatibleProvider(VLMProvider):
    """Provider for any OpenAI-compatible API (vLLM, OpenAI, etc.)."""

    def __init__(self, base_url: str, api_key: str, model_id: str) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id

    def generate(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
        except BadRequestError as exc:
            if "max_completion_tokens" in str(exc):
                # Older API (e.g. vLLM) doesn't support max_completion_tokens
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            else:
                raise
        except APIConnectionError:
            raise ConnectionError(
                f"Cannot connect to model '{self.model_id}' at {self.client.base_url}. "
                "Is the model server running?"
            )
        content = response.choices[0].message.content
        return (content or "").strip()
