from __future__ import annotations

from abc import ABC, abstractmethod


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
    def generate(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Send messages and return the model's text response."""
        ...
