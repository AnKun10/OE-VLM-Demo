"""File upload + retrieval service for the playground.

Storage layout: every file lives at IMAGES_DIR / "<uuid_hex>.<ext>". The
filename supplied by the client is preserved only in the response object
(`original_name`); the on-disk path is always UUID-derived.
"""
from __future__ import annotations

import io
import re
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

ALLOWED_MIME = {"image/png", "image/jpeg", "image/webp", "image/gif"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB
EXT_BY_MIME = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}
IMAGES_DIR = Path("images")
FILE_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


class StoredFile(BaseModel):
    """JSON output is camelCase (`originalName`) to match the frontend
    `AttachmentRef` type. Python field names stay snake_case.
    """
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: str
    url: str
    mime: str
    size: int
    original_name: str


def store_upload(upload: UploadFile) -> StoredFile:
    mime = (upload.content_type or "").lower()
    if mime not in ALLOWED_MIME:
        raise HTTPException(415, "Unsupported media type")

    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    size = len(data)
    if size == 0:
        raise HTTPException(400, "Not a valid image")
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large")

    try:
        Image.open(io.BytesIO(data)).verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise HTTPException(400, "Not a valid image")

    file_id = uuid.uuid4().hex
    ext = EXT_BY_MIME[mime]
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = IMAGES_DIR / f"{file_id}.tmp"
    final_path = IMAGES_DIR / f"{file_id}.{ext}"
    tmp_path.write_bytes(data)
    tmp_path.rename(final_path)

    return StoredFile(
        id=file_id,
        url=f"/api/files/{file_id}",
        mime=mime,
        size=size,
        original_name=upload.filename or "",
    )
