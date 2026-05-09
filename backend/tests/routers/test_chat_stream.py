"""Tests for /api/chat/stream SSE endpoint."""
from __future__ import annotations

import json
from typing import AsyncIterator
from unittest.mock import AsyncMock


def _parse_sse(body: bytes) -> list[dict]:
    """Parse SSE body bytes into a list of decoded JSON payloads."""
    out = []
    text = body.decode()
    for block in text.split("\n\n"):
        block = block.strip()
        if not block.startswith("data: "):
            continue
        payload = block[len("data: "):]
        out.append(json.loads(payload))
    return out


async def _make_async_iter(items):
    for it in items:
        yield it


def test_chat_stream_returns_text_event_stream(client, fake_manager):
    async def fake_stream(model_id, messages):
        for d in ["hello", " ", "world"]:
            yield d
    fake_manager.stream = fake_stream

    response = client.post(
        "/api/chat/stream",
        json={
            "model_id": "fake-vision",
            "messages": [{"role": "user", "text": "hi", "attachments": []}],
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.content)
    deltas = [e for e in events if "delta" in e and not e.get("done")]
    done = [e for e in events if e.get("done") is True]
    assert [e["delta"] for e in deltas] == ["hello", " ", "world"]
    assert len(done) == 1


def test_chat_stream_passes_messages_to_manager(client, fake_manager):
    captured = {}

    async def fake_stream(model_id, messages):
        captured["model_id"] = model_id
        captured["messages"] = messages
        yield "ok"
    fake_manager.stream = fake_stream

    response = client.post(
        "/api/chat/stream",
        json={
            "model_id": "fake-vision",
            "messages": [
                {"role": "user", "text": "hello", "attachments": []},
            ],
        },
    )
    assert response.status_code == 200
    assert captured["model_id"] == "fake-vision"
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
