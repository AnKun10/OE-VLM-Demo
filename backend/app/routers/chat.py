import base64
import io
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel
from PIL import Image

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


def resolve_image(url: str) -> Image.Image | None:
    """Resolve an image URL to a PIL Image. Supports local /images/ paths and base64 data URIs."""
    if url.startswith("data:"):
        try:
            _, data = url.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
        except Exception:
            return None
    if url.startswith("/images/"):
        path = Path("images") / url.removeprefix("/images/")
        if path.exists():
            return Image.open(path).convert("RGB")
    return None


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    message = request.message.strip()

    if not vlm_service.is_loaded():
        return ChatResponse(reply="Chatbot is currently unavailable!")

    # Build VLM prompt
    system_context = (
        "Ban la tro ly mua sam cua RunShop, cua hang giay chay bo. "
        "Tra loi bang tieng Viet, ngan gon va huu ich."
    )

    # Include recent history (last 4 turns max to keep prompt short)
    history_lines = []
    for msg in request.history[-4:]:
        role = "USER" if msg.role == "user" else "ASSISTANT"
        history_lines.append(f"{role}: {msg.content}")

    prompt_parts = [system_context] + history_lines + [message]
    prompt = "\n".join(prompt_parts)

    # Resolve image (use first one if provided)
    image = None
    for url in request.image_urls:
        image = resolve_image(url)
        if image is not None:
            break

    try:
        reply = vlm_service.generate_response(prompt, image=image)
    except Exception as exc:
        print(f"VLM generation error: {exc}")
        return ChatResponse(reply="Fail to response!")

    return ChatResponse(reply=reply)
