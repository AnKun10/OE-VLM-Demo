from fastapi import APIRouter, Depends
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.database import get_db
from app.services.qdrant_service import search_similar_products
from bson import ObjectId

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


class ChatProductRef(BaseModel):
    id: str
    name: str
    image_url: str | None = None


class ChatResponse(BaseModel):
    reply: str
    products: list[ChatProductRef] = []

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncIOMotorDatabase = Depends(get_db)):
    message = request.message.strip()

    # Try semantic search first, fall back to text search
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

    # Fall back to text search if semantic search returned nothing
    if not found_products:
        cursor = db.products.find(
            {"$text": {"$search": message}},
            {"score": {"$meta": "textScore"}},
        ).sort([("score", {"$meta": "textScore"})]).limit(3)
        async for doc in cursor:
            doc["id"] = str(doc.pop("_id"))
            found_products.append(doc)

    # Build reply
    if found_products:
        lines = ["Dựa trên yêu cầu của bạn, tôi gợi ý những sản phẩm sau:\n"]
        refs = []
        for p in found_products:
            store = p.get("store", "")
            category = p.get("category", "")
            meta = " - ".join(part for part in [store, category] if part)
            lines.append(f"• **{p['name']}**{f' - {meta}' if meta else ''}")
            refs.append(ChatProductRef(id=p["id"], name=p["name"], image_url=p.get("image_url")))

        lines.append(
            "\nBạn có muốn biết thêm chi tiết về sản phẩm nào không?"
        )
        reply = "\n".join(lines)
        return ChatResponse(reply=reply, products=refs)

    # Generic helpful response when no products matched
    lower = message.lower()
    if any(w in lower for w in ["giá", "rẻ", "khuyến mãi", "giảm"]):
        reply = (
            "Bạn có thể vào trang Sản phẩm và lọc theo cửa hàng hoặc danh mục để tìm sản phẩm phù hợp."
        )
    elif any(w in lower for w in ["trail", "địa hình", "núi", "đường mòn"]):
        reply = (
            "Giày trail running phù hợp cho địa hình phức tạp với đế bám tốt. "
            "HOKA Speedgoat và Salomon Speedcross là những lựa chọn phổ biến. "
            "Bạn có muốn tôi tìm kiếm thêm không?"
        )
    elif any(w in lower for w in ["road", "đường nhựa", "asphalt", "marathon"]):
        reply = (
            "Giày road running được thiết kế cho mặt đường phẳng với đệm tốt. "
            "Adidas, Nike và Puma đều có nhiều mẫu xuất sắc. "
            "Hãy cho tôi biết thêm yêu cầu của bạn!"
        )
    elif any(w in lower for w in ["size", "cỡ", "số"]):
        reply = (
            "Chúng tôi có sẵn các size từ 36 đến 47 tùy theo dòng sản phẩm. "
            "Thông thường nên chọn size lớn hơn 0.5 so với giày thường ngày khi mua giày chạy bộ."
        )
    else:
        reply = (
            "Xin chào! Tôi có thể giúp bạn:\n"
            "• Tìm giày chạy bộ phù hợp\n"
            "• So sánh các dòng sản phẩm\n"
            "• Tư vấn về size và fit\n"
            "• Thông tin khuyến mãi\n\n"
            "Bạn đang tìm kiếm loại giày nào?"
        )

    return ChatResponse(reply=reply)
