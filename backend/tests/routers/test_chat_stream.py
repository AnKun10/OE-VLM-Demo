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


def test_chat_stream_emits_connection_error_when_provider_fails_pre_first_chunk(
    client, fake_manager
):
    async def fake_stream(model_id, messages):
        raise ConnectionError("vLLM unreachable")
        yield  # unreachable but makes this an async generator
    fake_manager.stream = fake_stream

    response = client.post(
        "/api/chat/stream",
        json={"model_id": "fake-vision",
              "messages": [{"role": "user", "text": "hi", "attachments": []}]},
    )
    events = _parse_sse(response.content)
    assert len(events) == 1
    assert events[0]["error"] == "connection"


def test_chat_stream_emits_internal_error_after_partial_chunks(client, fake_manager):
    async def fake_stream(model_id, messages):
        yield "hi"
        yield " there"
        raise RuntimeError("kaboom")
    fake_manager.stream = fake_stream

    response = client.post(
        "/api/chat/stream",
        json={"model_id": "fake-vision",
              "messages": [{"role": "user", "text": "x", "attachments": []}]},
    )
    events = _parse_sse(response.content)
    deltas = [e for e in events if "delta" in e and not e.get("done")]
    errors = [e for e in events if e.get("error")]
    assert [e["delta"] for e in deltas] == ["hi", " there"]
    assert errors and errors[0]["error"] in {"internal", "bad_request"}


def test_chat_stream_emits_file_missing_when_attachment_unknown(client, fake_manager):
    """Unknown attachment id → SSE error file_missing, provider not called."""
    provider_called = {"n": 0}

    async def fake_stream(model_id, messages):
        provider_called["n"] += 1
        yield "should not happen"
    fake_manager.stream = fake_stream

    response = client.post(
        "/api/chat/stream",
        json={
            "model_id": "fake-vision",
            "messages": [{
                "role": "user", "text": "x",
                "attachments": [{"id": "0" * 32}],
            }],
        },
    )
    events = _parse_sse(response.content)
    assert events == [{"error": "file_missing",
                        "message": events[0]["message"]}] or any(
        e.get("error") == "file_missing" for e in events
    )
    assert provider_called["n"] == 0


def test_chat_stream_unknown_model_emits_bad_request(client, fake_manager):
    async def fake_stream(model_id, messages):
        raise RuntimeError("No VLM models are configured.")
        yield  # unreachable
    fake_manager.stream = fake_stream

    response = client.post(
        "/api/chat/stream",
        json={"model_id": "no-such-model",
              "messages": [{"role": "user", "text": "hi", "attachments": []}]},
    )
    events = _parse_sse(response.content)
    assert any(e.get("error") == "bad_request" for e in events)


def test_chat_stream_skips_yields_after_disconnect(client, fake_manager, monkeypatch):
    """Once Request.is_disconnected returns True, no more SSE events emitted."""
    yielded = []

    async def fake_stream(model_id, messages):
        for d in ["a", "b", "c", "d"]:
            yielded.append(d)
            yield d
    fake_manager.stream = fake_stream

    # Simulate disconnect after the second chunk is yielded.
    from starlette.requests import Request as StarletteRequest
    state = {"calls": 0}

    async def fake_is_disconnected(self):
        state["calls"] += 1
        return state["calls"] > 2

    monkeypatch.setattr(StarletteRequest, "is_disconnected", fake_is_disconnected)

    response = client.post(
        "/api/chat/stream",
        json={"model_id": "fake-vision",
              "messages": [{"role": "user", "text": "hi", "attachments": []}]},
    )
    events = _parse_sse(response.content)
    deltas = [e["delta"] for e in events if "delta" in e and not e.get("done")]
    # We may have stopped before all 4 deltas are SSE-emitted.
    assert len(deltas) <= 3
    assert "a" in deltas


def test_chat_stream_unicode_round_trip(client, fake_manager):
    """Vietnamese deltas survive JSON encoding."""
    async def fake_stream(model_id, messages):
        for d in ["Mèo ", "là ", "động vật."]:
            yield d
    fake_manager.stream = fake_stream

    response = client.post(
        "/api/chat/stream",
        json={"model_id": "fake-vision",
              "messages": [{"role": "user", "text": "tell me", "attachments": []}]},
    )
    events = _parse_sse(response.content)
    deltas = [e["delta"] for e in events if "delta" in e and not e.get("done")]
    assert "".join(deltas) == "Mèo là động vật."
