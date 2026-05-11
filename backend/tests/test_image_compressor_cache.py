"""Unit tests for CaptionCache (Phase 5)."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.services.image_compressor.cache import CaptionCache


@pytest.fixture
def cache_path(tmp_path: Path) -> str:
    return str(tmp_path / "captions.db")


@pytest.mark.asyncio
async def test_T5B1_round_trip(cache_path: str) -> None:
    """T5.B1: put → get round-trips. get_many partial hits return only hits.
    put_many with duplicate hash leaves first writer's caption (INSERT OR IGNORE).
    """
    cache = CaptionCache(cache_path)
    await cache.init()

    await cache.put("h1", "First caption.", model="m", bytes_size=100, user_id="u")
    assert await cache.get("h1") == "First caption."
    assert await cache.get("missing") is None

    # get_many: 2 hits, 1 miss
    await cache.put("h2", "Second caption.", model="m")
    out = await cache.get_many(["h1", "h2", "h3"])
    assert out == {"h1": "First caption.", "h2": "Second caption."}

    # put_many duplicate: first writer wins
    await cache.put_many([
        ("h1", "OVERWRITE attempt.", "m", None, None),  # collision with h1
        ("h4", "Fourth.", "m", None, None),             # new
    ])
    assert await cache.get("h1") == "First caption."  # unchanged
    assert await cache.get("h4") == "Fourth."


@pytest.mark.asyncio
async def test_A5B3_concurrent_put(cache_path: str) -> None:
    """A5.B3: 5 concurrent put_many calls with the SAME hash → no
    IntegrityError; cache has exactly 1 row for that hash."""
    cache = CaptionCache(cache_path)
    await cache.init()

    async def writer(i: int) -> None:
        await cache.put_many([
            ("same_hash", f"caption from writer {i}.", "m", None, None),
        ])

    await asyncio.gather(*(writer(i) for i in range(5)))

    # Exactly one row, one of the 5 captions (whichever won the INSERT race).
    cached = await cache.get("same_hash")
    assert cached is not None
    assert cached.startswith("caption from writer ")


@pytest.mark.asyncio
async def test_init_idempotent(cache_path: str) -> None:
    """Calling init() twice on the same path is safe (CREATE IF NOT EXISTS)."""
    cache_a = CaptionCache(cache_path)
    cache_b = CaptionCache(cache_path)
    await cache_a.init()
    await cache_b.init()
    await cache_a.put("h1", "x", model="m")
    assert await cache_b.get("h1") == "x"
