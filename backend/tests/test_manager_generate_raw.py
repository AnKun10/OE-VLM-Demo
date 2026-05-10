"""Test VLMManager.generate_raw — Phase 5 entry point for compressor calls."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from tests._helpers import make_async_stream_mock


@pytest.fixture
def real_manager():
    """A real VLMManager with one Qwen-VL provider mocked at the SDK boundary.
    We don't load YAML here; we hand-build a minimal config.
    """
    from app.models.vlm.manager import VLMManager
    from app.models.vlm.providers.openai_compatible import OpenAICompatibleProvider

    m = VLMManager()
    m.models["fake-model"] = {
        "id": "fake-model",
        "name": "Fake Model",
        "model_id": "fake-model",
        "system_prompt": "You are SYS.",
        "max_tokens": 256,
        "temperature": 0,
    }
    m.providers["fake-model"] = OpenAICompatibleProvider(
        base_url="http://fake/v1", api_key="none", model_id="fake-model",
    )
    m.default_model = "fake-model"
    return m


@pytest.mark.asyncio
async def test_generate_raw_skips_system_prompt(real_manager):
    """generate_raw must NOT prepend the model's system_prompt; caller's
    messages list is sent through verbatim."""
    sent_messages: list[list[dict]] = []

    def capture(*args, **kwargs):
        sent_messages.append(list(kwargs["messages"]))
        return make_async_stream_mock(["ok"])(*args, **kwargs)

    provider = real_manager.providers["fake-model"]
    with patch.object(provider.client.chat.completions, "create", side_effect=capture):
        result = await real_manager.generate_raw(
            "fake-model",
            [{"role": "user", "content": "hi"}],
            max_tokens=42, temperature=0.7,
        )

    assert result == "ok"
    assert len(sent_messages) == 1
    assert sent_messages[0] == [{"role": "user", "content": "hi"}]
    # NOTE: no {"role":"system","content":"You are SYS."} prepended.


@pytest.mark.asyncio
async def test_generate_raw_honors_max_tokens_override(real_manager):
    """Caller-supplied max_tokens reaches the SDK call (not the yaml default)."""
    captured_kwargs: list[dict] = []

    def capture(*args, **kwargs):
        captured_kwargs.append(kwargs)
        return make_async_stream_mock(["ok"])(*args, **kwargs)

    provider = real_manager.providers["fake-model"]
    with patch.object(provider.client.chat.completions, "create", side_effect=capture):
        await real_manager.generate_raw(
            "fake-model",
            [{"role": "user", "content": "hi"}],
            max_tokens=42, temperature=0.7,
        )

    # The SDK accepts either max_completion_tokens (modern) or max_tokens
    # (legacy fallback path). For the happy first call, our provider uses
    # max_completion_tokens=42 + temperature=0.7.
    assert captured_kwargs[0]["max_completion_tokens"] == 42
    assert captured_kwargs[0]["temperature"] == 0.7
