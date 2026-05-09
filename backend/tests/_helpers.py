# backend/tests/_helpers.py
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


def make_chunk(delta_text: str | None) -> MagicMock:
    """Build a MagicMock matching the ChatCompletionChunk shape used in
    `chunk.choices[0].delta.content`. Pass None to simulate function-call
    or empty deltas.
    """
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = delta_text
    return chunk


class FakeAsyncStream:
    """Async iterator mimicking openai.AsyncStream[ChatCompletionChunk]."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def make_async_stream_mock(deltas):
    """Return an AsyncMock that, when awaited, returns a FakeAsyncStream
    yielding ChatCompletionChunk-shaped objects with the given delta texts.

    Usage:
        mock_create = make_async_stream_mock(["hello", " world"])
        with patch.object(provider._client.chat.completions, "create", mock_create):
            ...
    """
    chunks = [make_chunk(d) for d in deltas]
    return AsyncMock(return_value=FakeAsyncStream(chunks))
