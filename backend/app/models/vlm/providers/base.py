from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator


class VLMProvider(ABC):
    """Base class for all VLM providers."""

    @classmethod
    def extra_kwargs_from_entry(cls, entry: dict) -> dict:
        """Return provider-specific constructor kwargs extracted from a YAML model entry.

        Override in subclasses to pull optional fields out of the entry dict.
        The default implementation returns an empty dict.
        """
        return {}

    @abstractmethod
    def stream(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        """Yield text deltas as they arrive from the upstream model.

        Implementations must be async generators (use `async def` + `yield`).
        """
        ...

    async def generate(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Convenience: collect stream into a single trimmed string."""
        chunks: list[str] = []
        async for delta in self.stream(messages, max_tokens, temperature):
            chunks.append(delta)
        return "".join(chunks).strip()
