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


# --- New streaming endpoint -------------------------------------------------

class Attachment(BaseModel):
    id: str


class ChatMessageWithAttachments(BaseModel):
    role: Literal["user", "assistant"]
    text: str = ""
    attachments: list[Attachment] = []


class ChatStreamRequest(BaseModel):
    messages: list[ChatMessageWithAttachments]
    model_id: str | None = None


def _sse_delta(delta: str) -> str:
    return f"data: {json.dumps({'delta': delta, 'done': False}, ensure_ascii=False)}\n\n"


def _sse_done() -> str:
    return f"data: {json.dumps({'delta': '', 'done': True})}\n\n"


def _sse_error(kind: str, message: str) -> str:
    return f"data: {json.dumps({'error': kind, 'message': message}, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def chat_stream(request: Request, body: ChatStreamRequest):
    manager = request.app.state.vlm_manager

    async def event_stream():
        try:
            openai_messages = build_openai_messages(body.messages)
            openai_messages = enforce_image_cap(openai_messages, max_images=4)
        except FileNotFoundError as exc:
            yield _sse_error("file_missing", str(exc))
            return

        try:
            async for delta in manager.stream(body.model_id, openai_messages):
                if await request.is_disconnected():
                    return
                yield _sse_delta(delta)
            yield _sse_done()
        except ConnectionError as exc:
            yield _sse_error("connection", str(exc))
        except BadRequestError as exc:
            yield _sse_error("bad_request", str(exc))
        except RuntimeError as exc:
            yield _sse_error("bad_request", str(exc))
        except Exception:
            traceback.print_exc()
            yield _sse_error("internal", "Internal error")

    return StreamingResponse(event_stream(), media_type="text/event-stream")
