import base64
import json
import traceback
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from openai import BadRequestError
from pydantic import BaseModel

from app.services.messages import build_openai_messages, enforce_image_cap

router = APIRouter(prefix="/api", tags=["chat"])


# --- Legacy non-streaming endpoint (kept for smoke script) -----------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    image_urls: list[str] = []
    model_id: str | None = None


class ChatResponse(BaseModel):
    reply: str


def resolve_image_url(url: str) -> str | None:
    """Resolve an image source to a data URI suitable for OpenAI-style
    image_url content. Supports base64 data URIs (passed through) and
    local /images/ paths (converted to base64 data URIs).
    """
    if url.startswith("data:"):
        return url
    if url.startswith("/images/"):
        path = Path("images") / url.removeprefix("/images/")
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            suffix = path.suffix.lstrip(".").lower()
            mime = {"jpg": "jpeg", "jpeg": "jpeg",
                    "png": "png", "webp": "webp"}.get(suffix, "jpeg")
            return f"data:image/{mime};base64,{data}"
    return None


@router.get("/models")
async def list_models(request: Request):
    manager = request.app.state.vlm_manager
    return {"models": manager.list_models()}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    manager = request.app.state.vlm_manager
    message = body.message.strip()

    image_data_url: str | None = None
    for url in body.image_urls:
        image_data_url = resolve_image_url(url)
        if image_data_url is not None:
            break

    messages: list[dict] = []
    for msg in body.history[-4:]:
        messages.append({"role": msg.role, "content": msg.content})

    if image_data_url is not None:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": message},
            ],
        })
    else:
        messages.append({"role": "user", "content": message})

    try:
        reply = await manager.generate(body.model_id, messages)
    except ConnectionError as exc:
        print(f"[chat] Connection error: {exc}")
        return ChatResponse(reply=f"Lỗi kết nối: {exc}")
    except Exception:
        traceback.print_exc()
        return ChatResponse(reply="Xin lỗi, không thể xử lý yêu cầu.")

    return ChatResponse(reply=reply)
