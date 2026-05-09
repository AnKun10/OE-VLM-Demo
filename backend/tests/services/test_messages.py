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
