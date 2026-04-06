import base64
from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["chat"])


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
    """Resolve an image source to a data URI suitable for OpenAI-style image_url content.

    Supports base64 data URIs (passed through) and local /images/ paths
    (converted to base64 data URIs).
    """
    if url.startswith("data:"):
        return url
    if url.startswith("/images/"):
        path = Path("images") / url.removeprefix("/images/")
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            suffix = path.suffix.lstrip(".").lower()
            mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(suffix, "jpeg")
            return f"data:image/{mime};base64,{data}"
    return None


@router.get("/models")
async def list_models(request: Request):
    manager = request.app.state.vlm_manager
    return {"models": manager.list_models()}


@router.post("/chat", response_model=ChatResponse)
def chat(request: Request, body: ChatRequest):
    manager = request.app.state.vlm_manager
    message = body.message.strip()

    # Resolve image data URI (use first valid one)
    image_data_url: str | None = None
    for url in body.image_urls:
        image_data_url = resolve_image_url(url)
        if image_data_url is not None:
            break

    # Build OpenAI-style messages (without system prompt — manager handles that)
    messages: list[dict] = []

    # Add conversation history (last 4 messages)
    for msg in body.history[-4:]:
        messages.append({"role": msg.role, "content": msg.content})

    # Build current user message content
    if image_data_url is not None:
        user_content: list[dict] = [
            {"type": "image_url", "image_url": {"url": image_data_url}},
            {"type": "text", "text": message},
        ]
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": message})

    try:
        reply = manager.generate(body.model_id, messages)
    except Exception:
        import traceback
        traceback.print_exc()
        return ChatResponse(reply="Xin lỗi, không thể xử lý yêu cầu.")

    return ChatResponse(reply=reply)
