"""Endpoint tests for /api/files."""
from __future__ import annotations

import io

from PIL import Image


def _png_bytes(size=(8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


def test_post_files_returns_camelcase(client, tmp_path, monkeypatch):
    from app.services import files as files_mod
    monkeypatch.setattr(files_mod, "IMAGES_DIR", tmp_path)

    data = _png_bytes()
    response = client.post(
        "/api/files",
        files={"file": ("hello.png", data, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "id" in body
    assert body["url"] == f"/api/files/{body['id']}"
    assert body["mime"] == "image/png"
    assert body["size"] == len(data)
    assert body["originalName"] == "hello.png"  # camelCase


def test_post_files_rejects_unsupported_mime(client, tmp_path, monkeypatch):
    from app.services import files as files_mod
    monkeypatch.setattr(files_mod, "IMAGES_DIR", tmp_path)
    response = client.post(
        "/api/files",
        files={"file": ("a.svg", b"<svg/>", "image/svg+xml")},
    )
    assert response.status_code == 415


def test_get_files_returns_image(client, tmp_path, monkeypatch):
    from app.services import files as files_mod
    monkeypatch.setattr(files_mod, "IMAGES_DIR", tmp_path)
    data = _png_bytes()
    fid = "f" * 32
    (tmp_path / f"{fid}.png").write_bytes(data)

    response = client.get(f"/api/files/{fid}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.content == data


def test_get_files_404_for_unknown_id(client, tmp_path, monkeypatch):
    from app.services import files as files_mod
    monkeypatch.setattr(files_mod, "IMAGES_DIR", tmp_path)
    response = client.get(f"/api/files/{'9' * 32}")
    assert response.status_code == 404


def test_get_files_400_for_invalid_id_format(client, tmp_path, monkeypatch):
    from app.services import files as files_mod
    monkeypatch.setattr(files_mod, "IMAGES_DIR", tmp_path)
    # Path-like id won't match the regex; FastAPI may also route it differently
    # so test a value that lands on the route but fails the regex.
    response = client.get("/api/files/" + "Z" * 32)  # uppercase fails
    assert response.status_code == 400


def test_get_files_finds_image_under_any_whitelisted_extension(client, tmp_path, monkeypatch):
    from app.services import files as files_mod
    monkeypatch.setattr(files_mod, "IMAGES_DIR", tmp_path)
    fid = "e" * 32
    (tmp_path / f"{fid}.webp").write_bytes(b"webp-bytes")

    response = client.get(f"/api/files/{fid}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/webp")
