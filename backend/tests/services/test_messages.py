"""Tests for services/messages.py."""
from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from app.services import files as files_mod
from app.services.messages import build_openai_messages


class _FakeMsg:
    def __init__(self, role, text, attachments=None):
        self.role = role
        self.text = text
        self.attachments = attachments or []


class _FakeAtt:
    def __init__(self, id):
        self.id = id


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_text_only_message_passes_through():
    msgs = [_FakeMsg("user", "hello")]
    out = build_openai_messages(msgs)
    assert out == [{"role": "user", "content": "hello"}]


def test_message_with_image_returns_content_array(tmp_path, monkeypatch):
    monkeypatch.setattr(files_mod, "IMAGES_DIR", tmp_path)
    fid = "1" * 32
    data = _png_bytes()
    (tmp_path / f"{fid}.png").write_bytes(data)

    msgs = [_FakeMsg("user", "what is this", attachments=[_FakeAtt(fid)])]
    out = build_openai_messages(msgs)
    assert len(out) == 1
    assert out[0]["role"] == "user"
    parts = out[0]["content"]
    assert len(parts) == 2
    assert parts[0]["type"] == "image_url"
    expected = f"data:image/png;base64,{base64.b64encode(data).decode()}"
    assert parts[0]["image_url"]["url"] == expected
    assert parts[1] == {"type": "text", "text": "what is this"}


def test_message_with_image_and_no_text_omits_text_part(tmp_path, monkeypatch):
    monkeypatch.setattr(files_mod, "IMAGES_DIR", tmp_path)
    fid = "2" * 32
    (tmp_path / f"{fid}.png").write_bytes(_png_bytes())

    msgs = [_FakeMsg("user", "", attachments=[_FakeAtt(fid)])]
    out = build_openai_messages(msgs)
    assert len(out[0]["content"]) == 1
    assert out[0]["content"][0]["type"] == "image_url"


def test_missing_attachment_raises_filenotfound(tmp_path, monkeypatch):
    monkeypatch.setattr(files_mod, "IMAGES_DIR", tmp_path)
    msgs = [_FakeMsg("user", "x", attachments=[_FakeAtt("3" * 32)])]
    with pytest.raises(FileNotFoundError):
        build_openai_messages(msgs)


def test_multiple_attachments_become_multiple_image_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(files_mod, "IMAGES_DIR", tmp_path)
    fid_a, fid_b = "a" * 32, "b" * 32
    (tmp_path / f"{fid_a}.png").write_bytes(_png_bytes())
    (tmp_path / f"{fid_b}.png").write_bytes(_png_bytes())

    msgs = [_FakeMsg("user", "two pics",
                     attachments=[_FakeAtt(fid_a), _FakeAtt(fid_b)])]
    out = build_openai_messages(msgs)
    image_parts = [p for p in out[0]["content"] if p["type"] == "image_url"]
    assert len(image_parts) == 2
    text_parts = [p for p in out[0]["content"] if p["type"] == "text"]
    assert text_parts == [{"type": "text", "text": "two pics"}]


from app.services.messages import PLACEHOLDER, enforce_image_cap


def _img_part(label="x"):
    return {"type": "image_url", "image_url": {"url": f"data:img,{label}"}}


def _text_part(text):
    return {"type": "text", "text": text}


def test_enforce_image_cap_under_limit_unchanged():
    messages = [
        {"role": "user", "content": [_img_part("a"), _text_part("hi")]},
        {"role": "assistant", "content": "ack"},
    ]
    out = enforce_image_cap(messages, max_images=4)
    assert out == messages


def test_enforce_image_cap_drops_oldest_first():
    messages = [
        {"role": "user", "content": [_img_part("a"), _text_part("first")]},
        {"role": "user", "content": [_img_part("b"), _img_part("c"),
                                      _img_part("d"), _img_part("e"),
                                      _text_part("now")]},
    ]
    out = enforce_image_cap(messages, max_images=4)
    # First image (oldest) should be replaced by placeholder; b/c/d/e remain.
    first_msg_parts = out[0]["content"]
    assert all(p.get("type") != "image_url" for p in first_msg_parts)
    assert any(p.get("type") == "text" and PLACEHOLDER in p["text"]
               for p in first_msg_parts)
    second_msg_parts = out[1]["content"]
    image_parts = [p for p in second_msg_parts if p["type"] == "image_url"]
    assert len(image_parts) == 4


def test_enforce_image_cap_replaces_lone_image_with_placeholder_string():
    """Single image_url + no text → content becomes placeholder string,
    not an empty array.
    """
    messages = [
        {"role": "user", "content": [_img_part("a")]},  # lone image
        {"role": "user", "content": [_img_part("b"), _img_part("c"),
                                      _img_part("d"), _img_part("e"),
                                      _text_part("ok")]},
    ]
    out = enforce_image_cap(messages, max_images=4)
    assert out[0]["content"] == PLACEHOLDER  # collapsed to string


def test_enforce_image_cap_eight_images_in_one_message_reduced_to_four():
    parts = [_img_part(str(i)) for i in range(8)] + [_text_part("end")]
    messages = [{"role": "user", "content": parts}]
    out = enforce_image_cap(messages, max_images=4)
    new_parts = out[0]["content"]
    image_parts = [p for p in new_parts if p["type"] == "image_url"]
    text_parts = [p for p in new_parts if p["type"] == "text"]
    assert len(image_parts) == 4
    # Placeholder text + original "end" — adjacent text segments may coalesce
    assert any(PLACEHOLDER in p["text"] for p in text_parts)
    assert any("end" in p["text"] for p in text_parts)


def test_enforce_image_cap_passes_through_text_only_messages():
    messages = [{"role": "user", "content": "no images here"}]
    out = enforce_image_cap(messages, max_images=4)
    assert out == messages
