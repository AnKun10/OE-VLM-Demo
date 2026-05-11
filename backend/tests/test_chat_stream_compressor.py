"""Integration tests: chat_stream + compressor engine (Phase 5)."""
from __future__ import annotations

import json

import pytest

from app.services.image_compressor.types import (
    CompressionResult, StatusEvent,
)


@pytest.fixture
def fake_engine():
    """A fake engine that scripts a fixed sequence of events for compress()."""
    class _FakeEngine:
        def __init__(self, *, events):
            self.events = events
            self.compress_calls = 0

        async def compress(self, messages):
            self.compress_calls += 1
            for ev in self.events:
                yield ev

    return _FakeEngine


def _parse_sse(body: bytes) -> list[dict]:
    """Parse SSE body into a list of decoded JSON payloads."""
    out = []
    for chunk in body.split(b"\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith(b"data:"):
            payload = chunk[5:].strip()
            out.append(json.loads(payload))
    return out


def test_T5B8_chat_stream_with_compressor(client, fake_manager, fake_engine):
    """T5.B8: 1 image in history → compressor emits status + thinking-log
    delta, then model deltas, then done."""
    from app.main import app

    img_url = "data:image/png;base64,iVBORw0KGgo="
    rewritten = [{"role": "user", "content": [
        {"type": "text", "text": "describe\n[Past image #1: a cat]"},
    ]}]

    app.state.compressor_engine = fake_engine(events=[
        StatusEvent(message="🖼️ Captioning 1 new image(s)..."),
        StatusEvent(message="✅ Compressor done", done=True),
        CompressionResult(
            messages=rewritten,
            thinking_md="<details><summary>🧠 reasoning</summary>...</details>",
        ),
    ])

    async def fake_stream(model_id, messages):
        # Compressor's rewritten messages reach the provider.
        assert messages == rewritten
        for delta in ["Hello ", "world."]:
            yield delta
    fake_manager.stream = fake_stream

    resp = client.post(
        "/api/chat/stream",
        json={
            "model_id": "fake-vision",
            "messages": [{
                "role": "user", "text": "describe",
                "attachments": [],
            }],
        },
    )
    assert resp.status_code == 200
    payloads = _parse_sse(resp.content)

    types_seen = []
    for p in payloads:
        if p.get("type") == "status":
            types_seen.append(("status", p.get("message"), p.get("done")))
        elif "delta" in p:
            types_seen.append(("delta", p["delta"], p.get("done")))
        else:
            types_seen.append(("error", p))

    assert types_seen[0] == ("status", "🖼️ Captioning 1 new image(s)...", False)
    assert types_seen[1] == ("status", "✅ Compressor done", True)
    # Next event is the thinking-log delta.
    assert types_seen[2][0] == "delta"
    assert "<details>" in types_seen[2][1]
    # Followed by model deltas.
    assert types_seen[3] == ("delta", "Hello ", False)
    assert types_seen[4] == ("delta", "world.", False)
    # Terminal done.
    assert types_seen[5] == ("delta", "", True)


def test_T5B9_chat_stream_no_images_fast_path(client, fake_manager, fake_engine):
    """T5.B9: 0 images → engine fast-path returns CompressionResult only;
    no status events, no thinking-log delta in stream."""
    from app.main import app
    app.state.compressor_engine = fake_engine(events=[
        CompressionResult(messages=[
            {"role": "user", "content": "hello"},
        ], thinking_md=""),
    ])

    async def fake_stream(model_id, messages):
        for delta in ["Hi ", "there."]:
            yield delta
    fake_manager.stream = fake_stream

    resp = client.post(
        "/api/chat/stream",
        json={
            "model_id": "fake-text",
            "messages": [{
                "role": "user", "text": "hello",
                "attachments": [],
            }],
        },
    )
    payloads = _parse_sse(resp.content)
    deltas = [p["delta"] for p in payloads if "delta" in p]
    statuses = [p for p in payloads if p.get("type") == "status"]

    # No status events, no thinking-log delta → just the model output + terminal.
    assert statuses == []
    assert deltas == ["Hi ", "there.", ""]


def test_A5_11_compressor_disabled_passthrough(client, fake_manager):
    """A5.11: when compressor_engine is None on app.state, chat_stream
    behaves exactly like Phase 4."""
    from app.main import app
    app.state.compressor_engine = None

    async def fake_stream(model_id, messages):
        for delta in ["Plain ", "answer."]:
            yield delta
    fake_manager.stream = fake_stream

    resp = client.post(
        "/api/chat/stream",
        json={
            "model_id": "fake-text",
            "messages": [{
                "role": "user", "text": "hi",
                "attachments": [],
            }],
        },
    )
    payloads = _parse_sse(resp.content)
    statuses = [p for p in payloads if p.get("type") == "status"]
    deltas = [p["delta"] for p in payloads if "delta" in p]

    assert statuses == []
    assert deltas == ["Plain ", "answer.", ""]
