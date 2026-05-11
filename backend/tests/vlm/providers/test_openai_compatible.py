"""Tests for OpenAICompatibleProvider async stream."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import APIConnectionError, BadRequestError

from app.models.vlm.providers.openai_compatible import OpenAICompatibleProvider
from tests._helpers import FakeAsyncStream, make_async_stream_mock, make_chunk


def _api_connection_error() -> APIConnectionError:
    return APIConnectionError(request=httpx.Request("POST", "http://fake"))


async def test_stream_yields_deltas_in_order():
    provider = OpenAICompatibleProvider(
        base_url="http://fake/v1", api_key="none", model_id="fake-model",
    )
    mock_create = make_async_stream_mock(["hello", " ", "world"])
    with patch.object(
        provider.client.chat.completions, "create", mock_create
    ):
        result = []
        async for delta in provider.stream(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10, temperature=0,
        ):
            result.append(delta)

    assert result == ["hello", " ", "world"]
    mock_create.assert_awaited_once()
    call_kwargs = mock_create.await_args.kwargs
    assert call_kwargs["stream"] is True
    assert call_kwargs["model"] == "fake-model"


async def test_stream_skips_none_delta_chunks():
    """Function-call placeholder chunks have delta.content = None — skip."""
    provider = OpenAICompatibleProvider(
        base_url="http://fake/v1", api_key="none", model_id="fake-model",
    )
    chunks = [make_chunk("hi"), make_chunk(None), make_chunk(" there")]
    fake_stream = FakeAsyncStream(chunks)
    mock_create = AsyncMock(return_value=fake_stream)
    with patch.object(
        provider.client.chat.completions, "create", mock_create
    ):
        result = [d async for d in provider.stream([], 10, 0)]

    assert result == ["hi", " there"]


async def test_stream_raises_connection_error_on_api_failure():
    provider = OpenAICompatibleProvider(
        base_url="http://fake/v1", api_key="none", model_id="fake-model",
    )
    mock_create = AsyncMock(side_effect=_api_connection_error())
    with patch.object(
        provider.client.chat.completions, "create", mock_create
    ):
        with pytest.raises(ConnectionError):
            async for _ in provider.stream([], 10, 0):
                pass


async def test_stream_falls_back_to_max_tokens_on_bad_request():
    """Older vLLM rejects max_completion_tokens; provider retries with max_tokens."""
    provider = OpenAICompatibleProvider(
        base_url="http://fake/v1", api_key="none", model_id="fake-model",
    )

    bad_req = BadRequestError(
        message="max_completion_tokens is not supported",
        response=httpx.Response(400, request=httpx.Request("POST", "http://fake")),
        body=None,
    )
    fake_stream = FakeAsyncStream([make_chunk("ok")])

    call_count = {"n": 0}
    async def fake_create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            assert "max_completion_tokens" in kwargs
            raise bad_req
        assert "max_tokens" in kwargs
        assert "max_completion_tokens" not in kwargs
        return fake_stream

    with patch.object(
        provider.client.chat.completions, "create", side_effect=fake_create
    ):
        result = [d async for d in provider.stream([], 50, 0)]

    assert result == ["ok"]
    assert call_count["n"] == 2
