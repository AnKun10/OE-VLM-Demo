import base64
import io
from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorDatabase
from PIL import Image
from bson import ObjectId

from app.database import get_db
from app.services.milvus_service import search_similar_products
from app.services import vlm_service

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    image_urls: list[str] = []


class ChatProductRef(BaseModel):
    id: str
    name: str
    image_url: str | None = None


class ChatResponse(BaseModel):
    reply: str
    products: list[ChatProductRef] = []


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
async def chat(request: ChatRequest, db: AsyncIOMotorDatabase = Depends(get_db)):
    message = request.message.strip()

    # --- Product search (semantic + text fallback) ---
    found_products = []
    try:
        ids = search_similar_products(message, top_k=3)
        if ids:
            valid_ids = [ObjectId(i) for i in ids if ObjectId.is_valid(i)]
            docs = await db.products.find({"_id": {"$in": valid_ids}}).to_list(length=len(valid_ids))
            doc_map = {str(doc["_id"]): doc for doc in docs}
            for product_id in ids:
                doc = doc_map.get(product_id)
                if not doc:
                    continue
                doc["id"] = str(doc.pop("_id"))
                found_products.append(doc)
    except Exception:
        pass

    if not found_products:
        try:
            cursor = db.products.find(
                {"$text": {"$search": message}},
                {"score": {"$meta": "textScore"}},
            ).sort([("score", {"$meta": "textScore"})]).limit(3)
            async for doc in cursor:
                doc["id"] = str(doc.pop("_id"))
                found_products.append(doc)
        except Exception:
            pass

    # Build product refs for response
    refs = [
        ChatProductRef(id=p["id"], name=p["name"], image_url=p.get("image_url"))
        for p in found_products
    ]

    # --- VLM response generation ---
    if not vlm_service.is_loaded():
        return ChatResponse(reply="❌ Chatbot is currently unavailable!")

    # Build VLM prompt
    system_context = (
        "Bạn là trợ lý mua sắm của RunShop, cửa hàng giày chạy bộ. "
        "Trả lời bằng tiếng Việt, ngắn gọn và hữu ích."
    )
    if found_products:
        product_info = "; ".join(
            f"{p['name']} ({p.get('store', '')}, {p.get('category', '')})"
            for p in found_products
        )
        system_context += f"\nSản phẩm liên quan trong cửa hàng: {product_info}"

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
        return ChatResponse(reply="❌ Fail to response")

    return ChatResponse(reply=reply, products=refs)
