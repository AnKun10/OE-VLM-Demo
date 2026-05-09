"""Models endpoint — list available VLM models with capabilities."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/api/models")
async def list_models(request: Request) -> dict[str, Any]:
    """Return list of available models with capabilities."""
    manager = request.app.state.vlm_manager
    return {"models": manager.list_models()}
