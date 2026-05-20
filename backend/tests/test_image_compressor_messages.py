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
    strip_assistant_thinking,
    strip_thinking_md,
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


# ---- selective rewrite (latest_image_turn_idx) -------------------------------

def test_rewrite_selective_silent_strips_older_image_turns() -> None:
    """When keep_idx=None AND latest_image_turn_idx is set, older image turns
    silently lose their images; latest image turn still gets captions."""
    msgs = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "uA"}},
            {"type": "text", "text": "describe this"},
        ]},
        {"role": "assistant", "content": "it is a mushroom"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "uB"}},
            {"type": "image_url", "image_url": {"url": "uC"}},
            {"type": "text", "text": "describe these"},
        ]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "con chó hình dáng như nào?"},
    ]
    captions = {"uA": "mushroom caption", "uB": "hair", "uC": "hands"}

    out = rewrite_messages(
        msgs, keep_idx=None, captions_by_url=captions,
        latest_image_turn_idx=2,
    )

    # Older image turn (msg 0): image stripped silently, text preserved,
    # NO caption appended.
    assert out[0]["content"] == [
        {"type": "text", "text": "describe this"},
    ]
    # Assistant unchanged.
    assert out[1]["content"] == "it is a mushroom"
    # Latest image turn (msg 2): images stripped, captions appended.
    assert out[2]["content"] == [
        {"type": "text", "text": "describe these\n[Past image #1: hair]\n[Past image #2: hands]"},
    ]
    # Text-only msgs untouched.
    assert out[3]["content"] == "ok"
    assert out[4]["content"] == "con chó hình dáng như nào?"


def test_rewrite_selective_image_only_older_turn_gets_placeholder() -> None:
    """An older image-only turn (no text) would become empty after silent
    strip; we insert a placeholder so downstream doesn't see content=[]."""
    msgs = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "uA"}},
        ]},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "uB"}},
            {"type": "text", "text": "ask"},
        ]},
    ]
    out = rewrite_messages(
        msgs, keep_idx=None, captions_by_url={"uA": "A", "uB": "B"},
        latest_image_turn_idx=1,
    )
    assert out[0]["content"] == [
        {"type": "text", "text": "[ảnh trước đó]"},
    ]
    assert out[1]["content"] == [
        {"type": "text", "text": "ask\n[Past image #1: B]"},
    ]


def test_rewrite_selective_no_op_when_keep_idx_set() -> None:
    """When keep_idx is set (router said keep, or kept_new_upload),
    latest_image_turn_idx must be ignored — existing behavior preserved."""
    msgs = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "uA"}},
            {"type": "text", "text": "first"},
        ]},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "uB"}},
            {"type": "text", "text": "second"},
        ]},
    ]
    out = rewrite_messages(
        msgs, keep_idx=1, captions_by_url={"uA": "A", "uB": "B"},
        latest_image_turn_idx=1,
    )
    # Msg 0 gets caption (NOT silently stripped); msg 1 untouched.
    assert out[0]["content"] == [
        {"type": "text", "text": "first\n[Past image #1: A]"},
    ]
    assert out[1]["content"] == msgs[1]["content"]


def test_rewrite_selective_no_op_when_latest_idx_none() -> None:
    """keep_idx=None and latest_image_turn_idx=None → all images become
    captions in their own turns (backwards-compat)."""
    msgs = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "uA"}},
            {"type": "text", "text": "first"},
        ]},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "uB"}},
            {"type": "text", "text": "second"},
        ]},
    ]
    out = rewrite_messages(
        msgs, keep_idx=None, captions_by_url={"uA": "A", "uB": "B"},
        latest_image_turn_idx=None,
    )
    assert out[0]["content"] == [
        {"type": "text", "text": "first\n[Past image #1: A]"},
    ]
    assert out[1]["content"] == [
        {"type": "text", "text": "second\n[Past image #1: B]"},
    ]


# ---- strip_thinking_md -------------------------------------------------------

def test_strip_thinking_md_removes_compressor_block() -> None:
    text = (
        "<details>\n"
        "<summary>🧠 Image compressor reasoning (4 ảnh, 0 caption mới, drop)</summary>\n"
        "\n"
        "**Step 1 — Image scan**\n"
        "- some content\n"
        "</details>\n\n"
        "Actual reply goes here."
    )
    out = strip_thinking_md(text)
    assert out.strip() == "Actual reply goes here."
    assert "🧠" not in out
    assert "<details>" not in out


def test_strip_thinking_md_handles_multiple_blocks() -> None:
    text = (
        "<details><summary>🧠 Image compressor reasoning (1 ảnh, kept)</summary>"
        "first thinking</details>\n"
        "reply 1\n"
        "<details><summary>🧠 Image compressor reasoning (2 ảnh, drop)</summary>"
        "second thinking</details>\n"
        "reply 2"
    )
    out = strip_thinking_md(text)
    assert "Image compressor" not in out
    assert "reply 1" in out
    assert "reply 2" in out


def test_strip_thinking_md_leaves_unrelated_details_alone() -> None:
    """Only the compressor's <details>...🧠 Image compressor...</details>
    is stripped. Other <details> blocks (e.g. user-written FAQ) survive."""
    text = (
        "<details><summary>User FAQ</summary>some legit content</details>\n"
        "reply"
    )
    out = strip_thinking_md(text)
    assert out == text


def test_strip_thinking_md_leaves_malformed_blocks() -> None:
    """Block without closing tag — don't crash, return as-is."""
    text = "<details><summary>🧠 Image compressor reasoning (broken)\nno close"
    out = strip_thinking_md(text)
    assert out == text


def test_strip_thinking_md_empty_input() -> None:
    assert strip_thinking_md("") == ""
    assert strip_thinking_md("no details here") == "no details here"


# ---- strip_assistant_thinking ------------------------------------------------

def test_strip_assistant_thinking_only_touches_assistant_strings() -> None:
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": (
            "<details><summary>🧠 Image compressor reasoning (x)</summary>"
            "stuff</details>\nreal reply"
        )},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "u"}},
            {"type": "text", "text": "see this"},
        ]},
        {"role": "assistant", "content": "plain reply, no thinking_md"},
        {"role": "system", "content": "<details><summary>🧠 Image compressor reasoning (x)</summary>stuff</details>"},
    ]
    out = strip_assistant_thinking(msgs)
    # User message preserved verbatim.
    assert out[0] == msgs[0]
    # Assistant with thinking_md: stripped.
    assert out[1]["role"] == "assistant"
    assert "<details>" not in out[1]["content"]
    assert "real reply" in out[1]["content"]
    # User with list content preserved verbatim.
    assert out[2] == msgs[2]
    # Assistant without thinking_md untouched.
    assert out[3] == msgs[3]
    # System role NOT touched (not assistant).
    assert out[4] == msgs[4]


def test_strip_assistant_thinking_no_mutation() -> None:
    msgs = [
        {"role": "assistant", "content": (
            "<details><summary>🧠 Image compressor reasoning</summary>"
            "x</details>\nreply"
        )},
    ]
    snapshot = msgs[0]["content"]
    out = strip_assistant_thinking(msgs)
    # Input must not be mutated; only the COPY in `out` is stripped.
    assert msgs[0]["content"] == snapshot
    assert out[0]["content"] != snapshot


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
