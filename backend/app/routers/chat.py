import base64
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from app.services import vlm_service

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    image_urls: list[str] = []


class ChatResponse(BaseModel):
    reply: str


def resolve_image_url(url: str) -> str | None:
    """Resolve an image source to a data URI suitable for vLLM image_url content.

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


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    message = request.message.strip()

    if not vlm_service.is_loaded():
        return ChatResponse(reply="Chatbot is currently unavailable!")

    # Resolve image data URI (use first valid one)
    image_data_url: str | None = None
    for url in request.image_urls:
        image_data_url = resolve_image_url(url)
        if image_data_url is not None:
            break

    # Build OpenAI-style messages for vLLM chat API
    system_context = (
        "Ban la tro ly mua sam cua RunShop, cua hang giay chay bo. "
        "Tra loi bang tieng Viet, ngan gon va huu ich."
    )

    messages: list[dict] = [{"role": "system", "content": system_context}]

    # Add conversation history (last 4 messages)
    for msg in request.history[-4:]:
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
        reply = vlm_service.generate_response(messages)
    except Exception as exc:
        print(f"VLM generation error: {exc}")
        return ChatResponse(reply="Fail to response!")

    return ChatResponse(reply=reply)
