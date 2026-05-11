"""Tests for VLMManager async API."""
from __future__ import annotations

from typing import AsyncIterator
from unittest.mock import patch

import pytest

from app.models.vlm.manager import VLMManager
from app.models.vlm.providers.base import VLMProvider


class _RecordingProvider(VLMProvider):
    """Records messages it received and yields fixed deltas."""

    def __init__(self, deltas):
        self._deltas = deltas
        self.received_messages: list[dict] | None = None
        self.received_max_tokens: int | None = None
        self.received_temperature: float | None = None

    async def stream(self, messages, max_tokens, temperature) -> AsyncIterator[str]:
        self.received_messages = messages
        self.received_max_tokens = max_tokens
        self.received_temperature = temperature
        for d in self._deltas:
            yield d


def _manager_with_provider(provider, *, system_prompt="", max_tokens=99, temperature=0.5):
    m = VLMManager()
    m.providers["m1"] = provider
    m.models["m1"] = {
        "id": "m1", "name": "Model 1",
        "system_prompt": system_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    m.default_model = "m1"
    return m


async def test_stream_prepends_system_prompt():
    provider = _RecordingProvider(["hi"])
    m = _manager_with_provider(provider, system_prompt="You are helpful.")
    deltas = [d async for d in m.stream("m1", [{"role": "user", "content": "ping"}])]
    assert deltas == ["hi"]
    assert provider.received_messages[0] == {"role": "system", "content": "You are helpful."}
    assert provider.received_messages[1] == {"role": "user", "content": "ping"}


async def test_stream_skips_empty_system_prompt():
    provider = _RecordingProvider(["hi"])
    m = _manager_with_provider(provider, system_prompt="   ")
    [_ async for _ in m.stream("m1", [{"role": "user", "content": "ping"}])]
    # No system message prepended
    assert provider.received_messages[0]["role"] == "user"


async def test_stream_passes_per_model_token_and_temperature():
    provider = _RecordingProvider(["x"])
    m = _manager_with_provider(provider, max_tokens=42, temperature=0.7)
    [_ async for _ in m.stream("m1", [{"role": "user", "content": "p"}])]
    assert provider.received_max_tokens == 42
    assert provider.received_temperature == 0.7


async def test_stream_falls_back_to_default_model_for_unknown_id():
    provider = _RecordingProvider(["x"])
    m = _manager_with_provider(provider)
    deltas = [d async for d in m.stream("does-not-exist", [{"role": "user", "content": "p"}])]
    assert deltas == ["x"]


async def test_stream_raises_when_no_models_configured():
    m = VLMManager()
    with pytest.raises(RuntimeError):
        async for _ in m.stream("any", []):
            pass


async def test_generate_collects_stream():
    provider = _RecordingProvider(["he", "llo"])
    m = _manager_with_provider(provider)
    result = await m.generate("m1", [{"role": "user", "content": "p"}])
    assert result == "hello"


def test_list_models_includes_capabilities_with_default_false():
    m = VLMManager()
    m.models["a"] = {"id": "a", "name": "A"}
    m.models["b"] = {"id": "b", "name": "B", "capabilities": {"vision": True}}
    listed = m.list_models()
    assert listed == [
        {"id": "a", "name": "A", "capabilities": {"vision": False}},
        {"id": "b", "name": "B", "capabilities": {"vision": True}},
    ]
