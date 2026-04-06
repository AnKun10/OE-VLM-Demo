from __future__ import annotations

from abc import ABC, abstractmethod


class VLMProvider(ABC):
    """Base class for all VLM providers."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Send messages and return the model's text response."""
        ...
