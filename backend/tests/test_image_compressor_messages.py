"""Unit tests for image_compressor.messages helpers (Phase 5)."""
from __future__ import annotations

import base64
import hashlib

import httpx
import pytest

from app.services.image_compressor.messages import (
    find_latest_image_turn,
    has_images,
    hash_image_url,
    iter_image_parts,
    rewrite_messages,
    text_of,
)


def _png_data_url() -> tuple[str, bytes]:
    """A 1x1 RGB PNG as data URL + its raw bytes."""
    raw = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452"
        "00000001000000010802000000907753"
        "de0000000c49444154789c6360606000"
        "0000040001f61738550000000049454e"
        "44ae426082"
    )
    b64 = base64.b64encode(raw).decode()
    return f"data:image/png;base64,{b64}", raw


# ---- T5.B2 iter / has_images / find_latest -----------------------------------

def test_T5B2_iter_image_parts_empty() -> None:
    assert list(iter_image_parts([])) == []


def test_T5B2_iter_image_parts_str_content_msg() -> None:
    msgs = [{"role": "user", "content": "hello"}]
    assert list(iter_image_parts(msgs)) == []


def test_T5B2_iter_image_parts_mixed() -> None:
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "u1"}},
            {"type": "image_url", "image_url": {"url": "u2"}},
        ]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "u3"}},
        ]},
    ]
    assert list(iter_image_parts(msgs)) == [
        (0, 1, "u1"), (0, 2, "u2"), (2, 0, "u3"),
    ]


def test_has_images() -> None:
    assert has_images({"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "u1"}},
    ]})
    assert not has_images({"role": "user", "content": "hello"})
    assert not has_images({"role": "user", "content": [
        {"type": "text", "text": "hi"},
    ]})


def test_find_latest_image_turn() -> None:
    msgs = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u1"}}]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "no images here"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u2"}}]},
        {"role": "user", "content": "also text only"},
    ]
    assert find_latest_image_turn(msgs) == 3
    assert find_latest_image_turn([]) is None
    assert find_latest_image_turn([{"role": "user", "content": "hi"}]) is None


def test_text_of() -> None:
    assert text_of({"role": "user", "content": "hi"}) == "hi"
    assert text_of({"role": "user", "content": [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "u"}},
    ]}) == "hello"
    assert text_of({"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "u"}},
    ]}) == ""


# ---- T5.B3 rewrite_messages --------------------------------------------------

def test_T5B3_rewrite_keeps_only_keep_idx() -> None:
    msgs = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "uA"}},
            {"type": "text", "text": "first"},
        ]},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "uB"}},
        ]},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "uC"}},
            {"type": "text", "text": "third"},
        ]},
    ]
    captions = {"uA": "alpha caption", "uB": "beta caption", "uC": "gamma caption"}

    out = rewrite_messages(msgs, keep_idx=2, captions_by_url=captions)

    # Msg 0 lost its image; text coalesced with [Past image #1: ...]
    assert out[0]["content"] == [
        {"type": "text", "text": "first\n[Past image #1: alpha caption]"},
    ]
    # Msg 1 was image-only; collapses to a string content with placeholder.
    assert out[1]["content"] == [
        {"type": "text", "text": "[Past image #1: beta caption]"},
    ]
    # Msg 2 (keep_idx) untouched.
    assert out[2]["content"] == msgs[2]["content"]


def test_T5B3_rewrite_drop_all_when_keep_idx_none() -> None:
    msgs = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "u1"}},
            {"type": "text", "text": "ask"},
        ]},
    ]
    out = rewrite_messages(msgs, keep_idx=None, captions_by_url={"u1": "the cat"})
    assert out[0]["content"] == [
        {"type": "text", "text": "ask\n[Past image #1: the cat]"},
    ]


def test_T5B3_rewrite_does_not_mutate_input() -> None:
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "u"}},
    ]}]
    snapshot = [dict(m) for m in msgs]
    rewrite_messages(msgs, keep_idx=None, captions_by_url={"u": "x"})
    assert msgs == snapshot


# ---- T5.B4 / A5.B4 hash_image_url --------------------------------------------

@pytest.mark.asyncio
async def test_T5B4_hash_data_url() -> None:
    url, raw = _png_data_url()
    h, body = await hash_image_url(url, fetch_base="http://x")
    assert h == hashlib.sha256(raw).hexdigest()
    assert body == raw


@pytest.mark.asyncio
async def test_T5B4_hash_bad_scheme() -> None:
    with pytest.raises(ValueError):
        await hash_image_url("ftp://x.png", fetch_base="http://x")


@pytest.mark.asyncio
async def test_T5B4_hash_empty_data_url() -> None:
    with pytest.raises(ValueError):
        await hash_image_url("data:image/png;base64,", fetch_base="http://x")


@pytest.mark.asyncio
async def test_A5B4_hash_relative_path(monkeypatch) -> None:
    """A5.B4: /api/files/<id> resolves against fetch_base via httpx."""
    _, raw = _png_data_url()

    class _Resp:
        def __init__(self, content: bytes) -> None:
            self.content = content
        def raise_for_status(self) -> None: pass

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def get(self, _url): return _Resp(raw)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    h, body = await hash_image_url(
        "/api/files/abc", fetch_base="http://127.0.0.1:8000",
    )
    assert h == hashlib.sha256(raw).hexdigest()
    assert body == raw
