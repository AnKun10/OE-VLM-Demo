"""Tests for the abstract VLMProvider base."""
from __future__ import annotations

from typing import AsyncIterator

import pytest

from app.models.vlm.providers.base import VLMProvider


class _FakeProvider(VLMProvider):
    def __init__(self, deltas):
        self._deltas = deltas

    async def stream(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        for d in self._deltas:
            yield d


async def test_generate_collects_stream_into_string():
    provider = _FakeProvider(["hello", " world"])
    result = await provider.generate(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        temperature=0,
    )
    assert result == "hello world"


async def test_generate_strips_trailing_whitespace():
    provider = _FakeProvider(["hi  ", "  "])
    result = await provider.generate([], 10, 0)
    assert result == "hi"
