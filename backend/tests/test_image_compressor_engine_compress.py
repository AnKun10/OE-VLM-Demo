"""Integration tests for ImageCompressorEngine.compress (Phase 5)."""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.image_compressor.cache import CaptionCache
from app.services.image_compressor.engine import ImageCompressorEngine
from app.services.image_compressor.types import (
    CompressionResult, StatusEvent,
)


def _png_data_url(byte: int) -> str:
    """Distinct 1-byte payloads → distinct hashes."""
    raw = bytes([byte])
    b64 = base64.b64encode(raw).decode()
    return f"data:image/png;base64,{b64}"


def _expected_hash(byte: int) -> str:
    return hashlib.sha256(bytes([byte])).hexdigest()


# ---- T5.B7 happy path with images --------------------------------------------

@pytest.mark.asyncio
async def test_T5B7_compress_with_images(tmp_path: Path) -> None:
    cache = CaptionCache(str(tmp_path / "c.db")); await cache.init()

    gen = AsyncMock(side_effect=[
        "Caption for image one.",
        "Caption for image two.",
        '{"need_images": false, "reason": "không liên quan ảnh"}',
    ])
    fake_mgr = MagicMock(); fake_mgr.generate_raw = gen
    engine = ImageCompressorEngine(
        cache=cache, vlm_manager=fake_mgr,
        caption_model_id="m", router_model_id="m",
    )

    msgs = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _png_data_url(1)}},
            {"type": "text", "text": "first turn"},
        ]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _png_data_url(2)}},
            {"type": "text", "text": "second turn"},
        ]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "third turn (text-only)"},
    ]

    events: list = []
    async for ev in engine.compress(msgs):
        events.append(ev)

    # At least one StatusEvent and exactly one terminal CompressionResult.
    statuses = [e for e in events if isinstance(e, StatusEvent)]
    results = [e for e in events if isinstance(e, CompressionResult)]
    assert len(statuses) >= 2          # captioning + routing + done
    assert any(s.done for s in statuses)
    assert len(results) == 1

    result = results[0]
    # Router said no-pixels → both image turns get `[Past image #1: ...]` inserts.
    text_0 = result.messages[0]["content"][-1]["text"]
    assert "[Past image #1: Caption for image one.]" in text_0
    text_2 = result.messages[2]["content"][-1]["text"]
    assert "[Past image #1: Caption for image two.]" in text_2
    # Last user turn untouched (no images).
    assert result.messages[4] == msgs[4]
    # Thinking md present and mentions reasoning.
    assert "🧠 Image compressor reasoning" in result.thinking_md
    assert "<details>" in result.thinking_md and "</details>" in result.thinking_md
    # CommonMark closes a type-6 HTML block (<details>) only at a BLANK
    # line. Without "</details>\n\n", the model response immediately after
    # is treated as part of the HTML block and inline markdown like
    # **bold** renders as literal asterisks.
    assert result.thinking_md.endswith("</details>\n\n"), (
        "thinking_md must end with </details>\\n\\n so the model response "
        "below is parsed as markdown, not raw HTML."
    )


# ---- T5.B7-keep: latest turn has images → keep_idx = latest ------------------

@pytest.mark.asyncio
async def test_T5B7_compress_latest_has_images(tmp_path: Path) -> None:
    """If the latest user turn carries images, we skip the router and keep
    those pixels (decision_label='kept new upload')."""
    cache = CaptionCache(str(tmp_path / "c.db")); await cache.init()

    gen = AsyncMock(return_value="Caption for older image.")
    fake_mgr = MagicMock(); fake_mgr.generate_raw = gen
    engine = ImageCompressorEngine(
        cache=cache, vlm_manager=fake_mgr,
        caption_model_id="m", router_model_id="m",
    )

    msgs = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _png_data_url(1)}},
            {"type": "text", "text": "old"},
        ]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _png_data_url(2)}},
            {"type": "text", "text": "new"},
        ]},
    ]

    events: list = [ev async for ev in engine.compress(msgs)]
    result = next(e for e in events if isinstance(e, CompressionResult))

    # Old turn stripped + caption insert; new turn UNTOUCHED.
    assert "[Past image #1: Caption for older image.]" in result.messages[0]["content"][-1]["text"]
    assert result.messages[2]["content"] == msgs[2]["content"]
    # We did NOT call the router (gen called once for the old image only).
    assert gen.call_count == 1


# ---- T5.B7-fast: 0 images → fast path ---------------------------------------

@pytest.mark.asyncio
async def test_T5B7_compress_no_images_fast_path(tmp_path: Path) -> None:
    cache = CaptionCache(str(tmp_path / "c.db")); await cache.init()
    gen = AsyncMock()  # should NEVER be called
    fake_mgr = MagicMock(); fake_mgr.generate_raw = gen
    engine = ImageCompressorEngine(
        cache=cache, vlm_manager=fake_mgr,
        caption_model_id="m", router_model_id="m",
    )

    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    events = [ev async for ev in engine.compress(msgs)]

    # Exactly one CompressionResult, zero StatusEvents.
    assert len(events) == 1
    assert isinstance(events[0], CompressionResult)
    assert events[0].messages == msgs
    assert events[0].thinking_md == ""
    gen.assert_not_called()


# ---- A5.B5 engine self-catch passthrough -------------------------------------

@pytest.mark.asyncio
async def test_A5B5_compress_self_catches_exceptions(tmp_path: Path) -> None:
    """Any exception inside compress is caught; result is a passthrough
    with thinking_md=''. Caller never sees the exception."""
    cache = CaptionCache(str(tmp_path / "c.db")); await cache.init()

    # Inject a bug: the cache.get_many on the engine raises.
    class BoomCache(CaptionCache):
        async def get_many(self, hashes):  # type: ignore[override]
            raise RuntimeError("DB unavailable")

    bad_cache = BoomCache(str(tmp_path / "c.db"))
    await bad_cache.init()

    gen = AsyncMock(return_value="caption")
    fake_mgr = MagicMock(); fake_mgr.generate_raw = gen
    engine = ImageCompressorEngine(
        cache=bad_cache, vlm_manager=fake_mgr,
        caption_model_id="m", router_model_id="m",
    )

    msgs = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _png_data_url(1)}},
        ]},
    ]
    events = [ev async for ev in engine.compress(msgs)]

    # Last event MUST be a passthrough CompressionResult.
    assert isinstance(events[-1], CompressionResult)
    assert events[-1].messages == msgs   # untouched
    assert events[-1].thinking_md == ""  # empty on failure
