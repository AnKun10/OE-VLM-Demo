"""Unit tests for engine helpers: caption_one, route, ensure_captions."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.image_compressor.cache import CaptionCache
from app.services.image_compressor.engine import ImageCompressorEngine


def _engine(cache: CaptionCache, *, generate_raw: AsyncMock) -> ImageCompressorEngine:
    """Engine with a fake VLMManager whose `generate_raw` is a script."""
    fake_mgr = MagicMock()
    fake_mgr.generate_raw = generate_raw
    return ImageCompressorEngine(
        cache=cache, vlm_manager=fake_mgr,
        caption_model_id="m", router_model_id="m",
        webui_internal_base="http://localhost:8000",
    )


# ---- T5.B5 caption_one happy path -------------------------------------------

@pytest.mark.asyncio
async def test_T5B5_caption_one_returns_trimmed(tmp_path: Path) -> None:
    cache = CaptionCache(str(tmp_path / "c.db")); await cache.init()
    gen = AsyncMock(return_value="  Một con mèo đen.  \n")
    engine = _engine(cache, generate_raw=gen)
    assert await engine.caption_one("data:image/png;base64,iVBORw==") == "Một con mèo đen."

    # Check it called manager with caption-shaped messages + max_tokens override.
    args, kwargs = gen.call_args
    assert kwargs["max_tokens"] == 80
    assert kwargs["messages"][0]["role"] == "system"
    assert "image captioner" in kwargs["messages"][0]["content"].lower()
    assert kwargs["messages"][1]["role"] == "user"


# ---- T5.B6 route happy + JSON parse fail-open --------------------------------

@pytest.mark.asyncio
async def test_T5B6_route_happy(tmp_path: Path) -> None:
    cache = CaptionCache(str(tmp_path / "c.db")); await cache.init()
    gen = AsyncMock(return_value='{"need_images": true, "reason": "câu hỏi đề cập ảnh"}')
    engine = _engine(cache, generate_raw=gen)

    decision, reason = await engine.route("Cái này là gì?", ["caption A"])
    assert decision is True
    assert reason == "câu hỏi đề cập ảnh"


@pytest.mark.asyncio
async def test_T5B6_route_non_json_fails_open_keep(tmp_path: Path) -> None:
    """When the router returns garbage, we fall back to router_failopen_keep
    (default True)."""
    cache = CaptionCache(str(tmp_path / "c.db")); await cache.init()
    gen = AsyncMock(return_value="LOL not JSON")
    engine = _engine(cache, generate_raw=gen)

    decision, reason = await engine.route("Q?", ["cap"])
    assert decision is True  # default failopen_keep
    assert "router failure" in reason


@pytest.mark.asyncio
async def test_T5B6_route_failopen_drop(tmp_path: Path) -> None:
    """If router_failopen_keep is False, parse error → drop images."""
    cache = CaptionCache(str(tmp_path / "c.db")); await cache.init()
    gen = AsyncMock(return_value="not json")
    fake_mgr = MagicMock(); fake_mgr.generate_raw = gen
    engine = ImageCompressorEngine(
        cache=cache, vlm_manager=fake_mgr,
        caption_model_id="m", router_model_id="m",
        router_failopen_keep=False,
    )

    decision, _ = await engine.route("Q?", ["cap"])
    assert decision is False


# ---- A5.B1 ensure_captions: per-image fail-open + cache fill -----------------

@pytest.mark.asyncio
async def test_A5B1_ensure_captions_partial_failure(tmp_path: Path) -> None:
    """ensure_captions: call 3 captions; one raises → it's OMITTED from the
    result; other two succeed and are written to cache."""
    cache = CaptionCache(str(tmp_path / "c.db")); await cache.init()

    call_count = {"n": 0}

    async def fake_generate_raw(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("boom")
        return f"caption #{call_count['n']}"

    fake_mgr = MagicMock(); fake_mgr.generate_raw = fake_generate_raw
    engine = ImageCompressorEngine(
        cache=cache, vlm_manager=fake_mgr,
        caption_model_id="m", router_model_id="m",
    )

    scanned = [
        (0, 0, "data:image/png;base64,A==", "h1", b"\x00"),
        (0, 1, "data:image/png;base64,B==", "h2", b"\x01"),
        (0, 2, "data:image/png;base64,C==", "h3", b"\x02"),
    ]
    out = await engine.ensure_captions(scanned)

    assert "data:image/png;base64,A==" in out
    assert "data:image/png;base64,C==" in out
    assert "data:image/png;base64,B==" not in out  # the failure is omitted

    # Cache filled with 2 captions (the survivors), keyed by hash:
    assert await cache.get("h1") is not None
    assert await cache.get("h2") is None
    assert await cache.get("h3") is not None


# ---- A5.B6 dedup-by-hash ------------------------------------------------------

@pytest.mark.asyncio
async def test_A5B6_ensure_captions_cache_hit(tmp_path: Path) -> None:
    """Hash-based dedup: 2 entries with same hash → captioner called once."""
    cache = CaptionCache(str(tmp_path / "c.db")); await cache.init()
    await cache.put("h_shared", "cached caption", model="m")

    gen = AsyncMock(return_value="UNCALLED")
    fake_mgr = MagicMock(); fake_mgr.generate_raw = gen
    engine = ImageCompressorEngine(
        cache=cache, vlm_manager=fake_mgr,
        caption_model_id="m", router_model_id="m",
    )

    scanned = [
        (0, 0, "url-A", "h_shared", b"\x00"),
        (1, 0, "url-B", "h_shared", b"\x00"),
    ]
    out = await engine.ensure_captions(scanned)
    assert out == {"url-A": "cached caption", "url-B": "cached caption"}
    gen.assert_not_called()
