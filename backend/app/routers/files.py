"""Routes for /api/files."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Response, UploadFile

from app.services.files import (
    FILE_ID_PATTERN,
    StoredFile,
    open_image_bytes,
    store_upload,
)

router = APIRouter(prefix="/api", tags=["files"])


@router.post("/files", response_model=StoredFile, response_model_by_alias=True)
async def upload_file(file: UploadFile = File(...)) -> StoredFile:
    return store_upload(file)


@router.get("/files/{file_id}")
async def get_file(file_id: str):
    if not FILE_ID_PATTERN.match(file_id):
        raise HTTPException(400, "Invalid file id")
    result = open_image_bytes(file_id)
    if result is None:
        raise HTTPException(404, "File not found")
    data, mime = result
    return Response(content=data, media_type=mime)
