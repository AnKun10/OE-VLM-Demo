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


def _build_fallback_reply(message: str) -> str:
    """Original hardcoded responses as fallback when VLM is not available."""
    lower = message.lower()
    if any(w in lower for w in ["giá", "rẻ", "khuyến mãi", "giảm"]):
        return "Bạn có thể vào trang Sản phẩm và lọc theo cửa hàng hoặc danh mục để tìm sản phẩm phù hợp."
    if any(w in lower for w in ["trail", "địa hình", "núi", "đường mòn"]):
        return (
            "Giày trail running phù hợp cho địa hình phức tạp với đế bám tốt. "
            "HOKA Speedgoat và Salomon Speedcross là những lựa chọn phổ biến. "
            "Bạn có muốn tôi tìm kiếm thêm không?"
        )
    if any(w in lower for w in ["road", "đường nhựa", "asphalt", "marathon"]):
        return (
            "Giày road running được thiết kế cho mặt đường phẳng với đệm tốt. "
            "Adidas, Nike và Puma đều có nhiều mẫu xuất sắc. "
            "Hãy cho tôi biết thêm yêu cầu của bạn!"
        )
    if any(w in lower for w in ["size", "cỡ", "số"]):
        return (
            "Chúng tôi có sẵn các size từ 36 đến 47 tùy theo dòng sản phẩm. "
            "Thông thường nên chọn size lớn hơn 0.5 so với giày thường ngày khi mua giày chạy bộ."
        )
    return (
        "Xin chào! Tôi có thể giúp bạn:\n"
        "• Tìm giày chạy bộ phù hợp\n"
        "• So sánh các dòng sản phẩm\n"
        "• Tư vấn về size và fit\n"
        "• Thông tin khuyến mãi\n\n"
        "Bạn đang tìm kiếm loại giày nào?"
    )


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
        # Fallback: hardcoded responses + product list
        if found_products:
            lines = ["Dựa trên yêu cầu của bạn, tôi gợi ý những sản phẩm sau:\n"]
            for p in found_products:
                store = p.get("store", "")
                category = p.get("category", "")
                meta = " - ".join(part for part in [store, category] if part)
                lines.append(f"• **{p['name']}**{f' - {meta}' if meta else ''}")
            lines.append("\nBạn có muốn biết thêm chi tiết về sản phẩm nào không?")
            return ChatResponse(reply="\n".join(lines), products=refs)
        return ChatResponse(reply=_build_fallback_reply(message))

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
        if found_products:
            lines = ["Dựa trên yêu cầu của bạn, tôi gợi ý những sản phẩm sau:\n"]
            for p in found_products:
                lines.append(f"• **{p['name']}**")
            reply = "\n".join(lines)
        else:
            reply = _build_fallback_reply(message)

    return ChatResponse(reply=reply, products=refs)
