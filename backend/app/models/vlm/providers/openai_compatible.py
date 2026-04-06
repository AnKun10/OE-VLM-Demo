# backend/app/models/vlm/providers/openai_compatible.py
from __future__ import annotations

from openai import OpenAI

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
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return (content or "").strip()
