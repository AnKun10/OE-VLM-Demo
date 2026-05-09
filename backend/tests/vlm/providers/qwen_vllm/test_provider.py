"""Async tests for QwenVLLMProvider."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import APIConnectionError, BadRequestError

from app.models.vlm.providers.qwen_vllm.provider import QwenVLLMProvider
from tests._helpers import FakeAsyncStream, make_async_stream_mock, make_chunk


def _api_connection_error() -> APIConnectionError:
    return APIConnectionError(request=httpx.Request("POST", "http://fake"))


async def test_stream_yields_deltas_on_success():
    provider = QwenVLLMProvider(
        base_url="http://fake/v1", api_key="none",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
    )
    mock_create = make_async_stream_mock(["hello", " world"])
    with patch.object(
        provider._client.chat.completions, "create", mock_create
    ):
        result = [d async for d in provider.stream(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10, temperature=0,
        )]

    assert result == ["hello", " world"]


async def test_stream_retries_once_on_pre_first_chunk_connection_error():
    provider = QwenVLLMProvider(
        base_url="http://fake/v1", api_key="none",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
    )

    fake_stream = FakeAsyncStream([make_chunk("ok")])
    call_count = {"n": 0}

    async def fake_create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _api_connection_error()
        return fake_stream

    with patch.object(
        provider._client.chat.completions, "create", side_effect=fake_create
    ), patch(
        "app.models.vlm.providers.qwen_vllm.provider.asyncio.sleep",
        new_callable=AsyncMock,
    ):
        result = [d async for d in provider.stream([], 10, 0)]

    assert result == ["ok"]
    assert call_count["n"] == 2


async def test_stream_raises_connection_error_after_retry_exhaustion():
    provider = QwenVLLMProvider(
        base_url="http://fake/v1", api_key="none",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
    )
    mock_create = AsyncMock(side_effect=_api_connection_error())
    with patch.object(
        provider._client.chat.completions, "create", mock_create
    ), patch(
        "app.models.vlm.providers.qwen_vllm.provider.asyncio.sleep",
        new_callable=AsyncMock,
    ):
        with pytest.raises(ConnectionError) as excinfo:
            async for _ in provider.stream([], 10, 0):
                pass

    assert "Qwen/Qwen3-VL-8B-Instruct" in str(excinfo.value)
    assert "http://fake/v1" in str(excinfo.value)


async def test_stream_applies_transforms_before_call():
    provider = QwenVLLMProvider(
        base_url="http://fake/v1", api_key="none",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        min_pixels=111, max_pixels=222,
    )
    captured: dict = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return FakeAsyncStream([make_chunk("ok")])

    with patch.object(
        provider._client.chat.completions, "create", side_effect=fake_create
    ):
        async for _ in provider.stream(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "x"}},
                    {"type": "text", "text": "see <image> this"},
                ],
            }],
            max_tokens=10, temperature=0,
        ):
            pass

    sent = captured["messages"]
    text_part = sent[0]["content"][1]
    img_part = sent[0]["content"][0]
    assert text_part["text"] == "see  this"
    assert img_part["image_url"]["min_pixels"] == 111
    assert img_part["image_url"]["max_pixels"] == 222
    assert captured["stream"] is True


async def test_stream_does_not_catch_bad_request_error():
    provider = QwenVLLMProvider(
        base_url="http://fake/v1", api_key="none",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
    )
    err = BadRequestError(
        message="bad",
        response=httpx.Response(400, request=httpx.Request("POST", "http://fake")),
        body=None,
    )
    mock_create = AsyncMock(side_effect=err)
    with patch.object(
        provider._client.chat.completions, "create", mock_create
    ):
        with pytest.raises(BadRequestError):
            async for _ in provider.stream([], 10, 0):
                pass


async def test_stream_does_not_retry_after_first_chunk():
    """Once a chunk has been yielded, mid-stream errors propagate as
    ConnectionError without retrying.
    """
    provider = QwenVLLMProvider(
        base_url="http://fake/v1", api_key="none",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
    )

    class StreamThenRaise(FakeAsyncStream):
        def __init__(self, chunks, raise_after):
            super().__init__(chunks)
            self._left = raise_after

        async def __anext__(self):
            if self._left == 0:
                raise _api_connection_error()
            self._left -= 1
            return await super().__anext__()

    call_count = {"n": 0}

    async def fake_create(**kwargs):
        call_count["n"] += 1
        return StreamThenRaise([make_chunk("a"), make_chunk("b")], raise_after=1)

    with patch.object(
        provider._client.chat.completions, "create", side_effect=fake_create
    ), patch(
        "app.models.vlm.providers.qwen_vllm.provider.asyncio.sleep",
        new_callable=AsyncMock,
    ):
        collected = []
        with pytest.raises(ConnectionError):
            async for d in provider.stream([], 10, 0):
                collected.append(d)

    assert collected == ["a"]
    assert call_count["n"] == 1  # NOT 2 — no retry post-first-chunk
