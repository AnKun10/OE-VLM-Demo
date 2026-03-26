from fastapi import APIRouter, Depends
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.database import get_db
from app.services.milvus_service import search_similar_products
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
    price: float


class ChatResponse(BaseModel):
    reply: str
    products: list[ChatProductRef] = []


def _format_price(price: float) -> str:
    return f"{int(price):,}đ".replace(",", ".")


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncIOMotorDatabase = Depends(get_db)):
    message = request.message.strip()

    # Try semantic search first, fall back to text search
    found_products = []
    try:
        ids = search_similar_products(message, top_k=3)
        if ids:
            valid_ids = [ObjectId(i) for i in ids if ObjectId.is_valid(i)]
            cursor = db.products.find({"_id": {"$in": valid_ids}})
            async for doc in cursor:
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
            price = p.get("price", 0)
            orig = p.get("original_price")
            price_str = _format_price(price)
            if orig and orig > price:
                discount = round((1 - price / orig) * 100)
                price_str = f"{price_str} (giảm {discount}%)"
            lines.append(f"• **{p['name']}** – {price_str}")
            refs.append(ChatProductRef(id=p["id"], name=p["name"], price=price))

        lines.append(
            "\nBạn có muốn biết thêm chi tiết về sản phẩm nào không?"
        )
        reply = "\n".join(lines)
        return ChatResponse(reply=reply, products=refs)

    # Generic helpful response when no products matched
    lower = message.lower()
    if any(w in lower for w in ["giá", "rẻ", "khuyến mãi", "giảm"]):
        reply = (
            "Hiện tại chúng tôi đang có nhiều ưu đãi hấp dẫn! "
            "Bạn có thể vào trang Sản phẩm và lọc theo mức giá để tìm đôi giày phù hợp với ngân sách của mình."
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
