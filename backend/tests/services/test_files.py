"""Tests for services/files.py."""
from __future__ import annotations

import io

import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

from app.services import files as files_mod


def _png_bytes(size=(8, 8), color=(255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(size=(8, 8), color=(0, 255, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def _make_upload(filename: str, mime: str, data: bytes) -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(data),
                      headers={"content-type": mime})


def test_store_upload_writes_png(tmp_path, monkeypatch):
    monkeypatch.setattr(files_mod, "IMAGES_DIR", tmp_path)
    data = _png_bytes()
    upload = _make_upload("foo.png", "image/png", data)

    stored = files_mod.store_upload(upload)

    assert stored.id  # uuid hex
    assert stored.url == f"/api/files/{stored.id}"
    assert stored.mime == "image/png"
    assert stored.size == len(data)
    assert stored.original_name == "foo.png"

    written = (tmp_path / f"{stored.id}.png").read_bytes()
    assert written == data


def test_store_upload_assigns_jpg_extension_for_jpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(files_mod, "IMAGES_DIR", tmp_path)
    upload = _make_upload("photo.jpeg", "image/jpeg", _jpeg_bytes())
    stored = files_mod.store_upload(upload)
    assert (tmp_path / f"{stored.id}.jpg").exists()


def test_store_upload_rejects_unknown_mime(tmp_path, monkeypatch):
    monkeypatch.setattr(files_mod, "IMAGES_DIR", tmp_path)
    upload = _make_upload("a.svg", "image/svg+xml", b"<svg/>")
    with pytest.raises(HTTPException) as exc:
        files_mod.store_upload(upload)
    assert exc.value.status_code == 415


def test_store_upload_rejects_zero_byte(tmp_path, monkeypatch):
    monkeypatch.setattr(files_mod, "IMAGES_DIR", tmp_path)
    upload = _make_upload("empty.png", "image/png", b"")
    with pytest.raises(HTTPException) as exc:
        files_mod.store_upload(upload)
    assert exc.value.status_code == 400


def test_store_upload_rejects_oversized(tmp_path, monkeypatch):
    monkeypatch.setattr(files_mod, "IMAGES_DIR", tmp_path)
    monkeypatch.setattr(files_mod, "MAX_UPLOAD_BYTES", 1024)
    upload = _make_upload("big.png", "image/png", b"\x00" * 2048)
    with pytest.raises(HTTPException) as exc:
        files_mod.store_upload(upload)
    assert exc.value.status_code == 413


def test_store_upload_rejects_spoofed_mime(tmp_path, monkeypatch):
    """Content-Type says PNG, body is plain text — PIL.verify() fails."""
    monkeypatch.setattr(files_mod, "IMAGES_DIR", tmp_path)
    upload = _make_upload("trick.png", "image/png", b"hello, not an image")
    with pytest.raises(HTTPException) as exc:
        files_mod.store_upload(upload)
    assert exc.value.status_code == 400


def test_store_upload_rejects_corrupt_jpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(files_mod, "IMAGES_DIR", tmp_path)
    truncated = _jpeg_bytes()[:32]  # cut JPEG mid-header
    upload = _make_upload("corrupt.jpg", "image/jpeg", truncated)
    with pytest.raises(HTTPException) as exc:
        files_mod.store_upload(upload)
    assert exc.value.status_code == 400


def test_store_upload_strips_path_traversal_filename(tmp_path, monkeypatch):
    """The on-disk path uses the UUID; filename never traverses."""
    monkeypatch.setattr(files_mod, "IMAGES_DIR", tmp_path)
    upload = _make_upload("../../etc/passwd", "image/png", _png_bytes())
    stored = files_mod.store_upload(upload)
    # No file outside tmp_path
    expected_path = tmp_path / f"{stored.id}.png"
    assert expected_path.exists()
    # Nothing got written to a parent
    assert not (tmp_path.parent / "etc").exists()


def test_open_image_bytes_returns_bytes_and_mime(tmp_path, monkeypatch):
    monkeypatch.setattr(files_mod, "IMAGES_DIR", tmp_path)
    data = _png_bytes()
    fid = "a" * 32
    (tmp_path / f"{fid}.png").write_bytes(data)

    result = files_mod.open_image_bytes(fid)
    assert result is not None
    bytes_, mime = result
    assert bytes_ == data
    assert mime == "image/png"


def test_open_image_bytes_finds_jpeg_extension(tmp_path, monkeypatch):
    monkeypatch.setattr(files_mod, "IMAGES_DIR", tmp_path)
    data = _jpeg_bytes()
    fid = "b" * 32
    (tmp_path / f"{fid}.jpg").write_bytes(data)

    result = files_mod.open_image_bytes(fid)
    assert result is not None
    _, mime = result
    assert mime == "image/jpeg"


def test_open_image_bytes_returns_none_for_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(files_mod, "IMAGES_DIR", tmp_path)
    assert files_mod.open_image_bytes("c" * 32) is None


def test_open_image_bytes_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(files_mod, "IMAGES_DIR", tmp_path)
    assert files_mod.open_image_bytes("../etc/passwd") is None
    assert files_mod.open_image_bytes("foo/bar") is None
    assert files_mod.open_image_bytes("ABCDEF12" * 4) is None  # uppercase
    assert files_mod.open_image_bytes("a" * 31) is None  # too short
    assert files_mod.open_image_bytes("a" * 33) is None  # too long
