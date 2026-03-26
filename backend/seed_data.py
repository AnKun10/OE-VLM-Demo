"""
Seed script: populates MongoDB with sample shoe products and indexes them in Milvus.
Run from the backend/ directory:
    python seed_data.py
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
from app.database import connect_milvus
from app.services.milvus_service import upsert_product_embedding

PRODUCTS = [
    # HOKA
    {
        "name": "Giày Chạy Trail Nữ HOKA Speedgoat 6 - Đỏ",
        "brand": "HOKA",
        "category": "Trail Running",
        "description": "HOKA Speedgoat 6 là đôi giày trail running hàng đầu với đế ngoài Vibram® Megagrip cung cấp độ bám tốt trên mọi địa hình. Phần đệm CMEVA mềm mại mang lại cảm giác thoải mái cho những chặng đường dài.",
        "price": 2000000,
        "original_price": 3999000,
        "colors": [
            {"name": "Đỏ", "hex": "#8B1A1A"},
            {"name": "Đen", "hex": "#1C1C1C"},
            {"name": "Trắng", "hex": "#F5F5F5"},
        ],
        "sizes": ["36", "37", "38", "39", "40", "41"],
        "images": [
            "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=600&q=80",
            "https://images.unsplash.com/photo-1539185441755-769473a23570?w=600&q=80",
        ],
        "tags": ["trail", "running", "women", "hoka", "speedgoat", "vibram"],
        "is_new": False,
        "in_stock": True,
        "stock_qty": 45,
        "rating": 4.8,
        "review_count": 124,
    },
    {
        "name": "Giày Chạy Trail Nam HOKA Speedgoat 6 - Trắng",
        "brand": "HOKA",
        "category": "Trail Running",
        "description": "HOKA Speedgoat 6 dành cho nam với thiết kế năng động. Đế Vibram® Megagrip bám dính vượt trội, phần upper thoáng khí giúp bàn chân luôn khô ráo khi chinh phục các cung đường trail.",
        "price": 2399000,
        "original_price": 3999000,
        "colors": [
            {"name": "Trắng", "hex": "#F5F5F5"},
            {"name": "Cam", "hex": "#FF6B35"},
        ],
        "sizes": ["39", "40", "41", "42", "43", "44"],
        "images": [
            "https://images.unsplash.com/photo-1606107557195-0e29a4b5b4aa?w=600&q=80",
            "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=600&q=80",
        ],
        "tags": ["trail", "running", "men", "hoka", "speedgoat", "vibram"],
        "is_new": False,
        "in_stock": True,
        "stock_qty": 38,
        "rating": 4.7,
        "review_count": 98,
    },
    {
        "name": "Giày Chạy Bộ Nam HOKA Clifton 9 - Xanh Navy",
        "brand": "HOKA",
        "category": "Road Running",
        "description": "HOKA Clifton 9 tiếp tục truyền thống với phần đệm siêu nhẹ và mềm mại. Lý tưởng cho các buổi chạy bộ hàng ngày, cung cấp sự thoải mái tối đa từ bước đầu đến bước cuối.",
        "price": 3500000,
        "original_price": None,
        "colors": [
            {"name": "Xanh Navy", "hex": "#001F5B"},
            {"name": "Xám", "hex": "#808080"},
        ],
        "sizes": ["40", "41", "42", "43", "44"],
        "images": [
            "https://images.unsplash.com/photo-1491553895911-0055eca6402d?w=600&q=80",
            "https://images.unsplash.com/photo-1460353581641-37baddab0fa2?w=600&q=80",
        ],
        "tags": ["road", "running", "men", "hoka", "clifton", "cushion"],
        "is_new": True,
        "in_stock": True,
        "stock_qty": 20,
        "rating": 4.9,
        "review_count": 67,
    },
    # PUMA
    {
        "name": "Giày Chạy Bộ Nam Puma Deviate Nitro Elite 4 - Xanh Mint",
        "brand": "PUMA",
        "category": "Road Running",
        "description": "Puma Deviate Nitro Elite 4 là đỉnh cao công nghệ chạy bộ của PUMA với tấm carbon fiber plate và đệm NITRO Elite foam siêu nhẹ. Thiết kế cho những vận động viên thi đấu chuyên nghiệp.",
        "price": 5950000,
        "original_price": None,
        "colors": [
            {"name": "Xanh Mint", "hex": "#98D8C8"},
            {"name": "Vàng", "hex": "#FFD700"},
        ],
        "sizes": ["39", "40", "41", "42", "43", "44", "45"],
        "images": [
            "https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?w=600&q=80",
            "https://images.unsplash.com/photo-1587563871167-1ee9c731aefb?w=600&q=80",
        ],
        "tags": ["road", "running", "men", "puma", "nitro", "carbon", "race"],
        "is_new": True,
        "in_stock": True,
        "stock_qty": 15,
        "rating": 4.6,
        "review_count": 43,
    },
    {
        "name": "Giày Chạy Bộ Nữ Puma Velocity Nitro 3 - Hồng",
        "brand": "PUMA",
        "category": "Road Running",
        "description": "Puma Velocity Nitro 3 dành cho nữ với công nghệ NITRO foam nhẹ và đàn hồi. Phần upper knit thoáng khí ôm sát bàn chân, phù hợp cho chạy bộ hàng ngày.",
        "price": 2800000,
        "original_price": 3500000,
        "colors": [
            {"name": "Hồng", "hex": "#FFB6C1"},
            {"name": "Trắng", "hex": "#FFFFFF"},
        ],
        "sizes": ["36", "37", "38", "39", "40"],
        "images": [
            "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=600&q=80",
            "https://images.unsplash.com/photo-1608231387042-66d1773070a5?w=600&q=80",
        ],
        "tags": ["road", "running", "women", "puma", "nitro", "daily"],
        "is_new": False,
        "in_stock": True,
        "stock_qty": 30,
        "rating": 4.4,
        "review_count": 56,
    },
    # ADIDAS
    {
        "name": "Giày Chạy Bộ Nam Adidas Supernova Rise 3 - Xanh Dương",
        "brand": "ADIDAS",
        "category": "Road Running",
        "description": "Adidas Supernova Rise 3 với lớp đệm DREAMSTRIKE+ mang lại cảm giác đàn hồi và êm ái. Thiết kế hiện đại phù hợp cả cho chạy bộ lẫn hoạt động hàng ngày.",
        "price": 3800000,
        "original_price": None,
        "colors": [
            {"name": "Xanh Dương", "hex": "#1E3A8A"},
            {"name": "Xanh Lam", "hex": "#3B82F6"},
            {"name": "Xanh Ngọc", "hex": "#0D9488"},
        ],
        "sizes": ["39", "40", "41", "42", "43", "44"],
        "images": [
            "https://images.unsplash.com/photo-1608231387042-66d1773070a5?w=600&q=80",
            "https://images.unsplash.com/photo-1606107557195-0e29a4b5b4aa?w=600&q=80",
        ],
        "tags": ["road", "running", "men", "adidas", "supernova", "dreamstrike"],
        "is_new": True,
        "in_stock": True,
        "stock_qty": 25,
        "rating": 4.5,
        "review_count": 82,
    },
    {
        "name": "Giày Chạy Bộ Nam Adidas Ultraboost 22 - Đen",
        "brand": "ADIDAS",
        "category": "Road Running",
        "description": "Adidas Ultraboost 22 huyền thoại với đế BOOST cho năng lượng hoàn trả tốt nhất. Upper PRIMEKNIT ôm sát, Torsion System hỗ trợ vòm bàn chân tối ưu.",
        "price": 4500000,
        "original_price": 5500000,
        "colors": [
            {"name": "Đen", "hex": "#000000"},
            {"name": "Trắng", "hex": "#FFFFFF"},
        ],
        "sizes": ["39", "40", "41", "42", "43", "44", "45"],
        "images": [
            "https://images.unsplash.com/photo-1539185441755-769473a23570?w=600&q=80",
            "https://images.unsplash.com/photo-1491553895911-0055eca6402d?w=600&q=80",
        ],
        "tags": ["road", "running", "men", "adidas", "ultraboost", "boost"],
        "is_new": False,
        "in_stock": True,
        "stock_qty": 40,
        "rating": 4.7,
        "review_count": 215,
    },
    # NIKE
    {
        "name": "Giày Chạy Bộ Nam Nike Pegasus 41 - Đỏ",
        "brand": "NIKE",
        "category": "Road Running",
        "description": "Nike Air Zoom Pegasus 41 tiếp tục là người bạn đồng hành tin cậy cho mọi vận động viên. Túi khí Air Zoom ở mũi và gót giày tạo độ bật nhảy tuyệt vời.",
        "price": 3200000,
        "original_price": None,
        "colors": [
            {"name": "Đỏ", "hex": "#CC0000"},
            {"name": "Đen", "hex": "#111111"},
        ],
        "sizes": ["39", "40", "41", "42", "43", "44"],
        "images": [
            "https://images.unsplash.com/photo-1460353581641-37baddab0fa2?w=600&q=80",
            "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=600&q=80",
        ],
        "tags": ["road", "running", "men", "nike", "pegasus", "zoom", "air"],
        "is_new": True,
        "in_stock": True,
        "stock_qty": 22,
        "rating": 4.6,
        "review_count": 178,
    },
    {
        "name": "Giày Chạy Bộ Nữ Nike Vomero 17 - Tím",
        "brand": "NIKE",
        "category": "Road Running",
        "description": "Nike Vomero 17 dành cho nữ với hệ thống đệm ZoomX foam mang lại cảm giác mây đỡ chân. Thiết kế tối giản, màu tím thanh lịch phù hợp mọi hoàn cảnh.",
        "price": 3800000,
        "original_price": 4500000,
        "colors": [
            {"name": "Tím", "hex": "#7C3AED"},
            {"name": "Hồng Nhạt", "hex": "#FBCFE8"},
        ],
        "sizes": ["36", "37", "38", "39", "40", "41"],
        "images": [
            "https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?w=600&q=80",
            "https://images.unsplash.com/photo-1606107557195-0e29a4b5b4aa?w=600&q=80",
        ],
        "tags": ["road", "running", "women", "nike", "vomero", "zoomx", "cushion"],
        "is_new": False,
        "in_stock": True,
        "stock_qty": 18,
        "rating": 4.5,
        "review_count": 91,
    },
    # BROOKS
    {
        "name": "Giày Chạy Bộ Nam Brooks Ghost 16 - Xanh Lá",
        "brand": "BROOKS",
        "category": "Road Running",
        "description": "Brooks Ghost 16 là lựa chọn đáng tin cậy cho người chạy bộ trung tính. Công nghệ DNA LOFT v3 mang lại cảm giác đệm nhẹ nhàng trong suốt hành trình.",
        "price": 3600000,
        "original_price": None,
        "colors": [
            {"name": "Xanh Lá", "hex": "#166534"},
            {"name": "Xám Bạc", "hex": "#9CA3AF"},
        ],
        "sizes": ["40", "41", "42", "43", "44", "45"],
        "images": [
            "https://images.unsplash.com/photo-1587563871167-1ee9c731aefb?w=600&q=80",
            "https://images.unsplash.com/photo-1460353581641-37baddab0fa2?w=600&q=80",
        ],
        "tags": ["road", "running", "men", "brooks", "ghost", "neutral", "dna"],
        "is_new": True,
        "in_stock": True,
        "stock_qty": 12,
        "rating": 4.8,
        "review_count": 54,
    },
    {
        "name": "Giày Chạy Bộ Nữ Brooks Adrenaline GTS 24 - Hồng Cam",
        "brand": "BROOKS",
        "category": "Road Running",
        "description": "Brooks Adrenaline GTS 24 với GuideRails® hỗ trợ giữ cho đầu gối và hông ở vị trí tốt nhất. Lý tưởng cho người có vòm bàn chân thấp hoặc bàn chân lật vào.",
        "price": 3400000,
        "original_price": 4200000,
        "colors": [
            {"name": "Hồng Cam", "hex": "#FB923C"},
            {"name": "Xanh Dương", "hex": "#2563EB"},
        ],
        "sizes": ["36", "37", "38", "39", "40"],
        "images": [
            "https://images.unsplash.com/photo-1608231387042-66d1773070a5?w=600&q=80",
            "https://images.unsplash.com/photo-1539185441755-769473a23570?w=600&q=80",
        ],
        "tags": ["road", "running", "women", "brooks", "adrenaline", "support", "stability"],
        "is_new": False,
        "in_stock": True,
        "stock_qty": 27,
        "rating": 4.6,
        "review_count": 73,
    },
    # ASICS
    {
        "name": "Giày Chạy Bộ Nam ASICS Gel-Kayano 31 - Trắng",
        "brand": "ASICS",
        "category": "Road Running",
        "description": "ASICS Gel-Kayano 31 - mẫu giày stability hàng đầu với hệ thống GEL™ ở gót giày. FF BLAST™ PLUS ECO foam nhẹ hơn 11% so với phiên bản trước.",
        "price": 4200000,
        "original_price": None,
        "colors": [
            {"name": "Trắng", "hex": "#F9FAFB"},
            {"name": "Đen Xanh", "hex": "#1E293B"},
        ],
        "sizes": ["39", "40", "41", "42", "43", "44"],
        "images": [
            "https://images.unsplash.com/photo-1491553895911-0055eca6402d?w=600&q=80",
            "https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?w=600&q=80",
        ],
        "tags": ["road", "running", "men", "asics", "gel-kayano", "stability", "gel"],
        "is_new": True,
        "in_stock": True,
        "stock_qty": 16,
        "rating": 4.7,
        "review_count": 88,
    },
    {
        "name": "Giày Chạy Bộ Nữ ASICS Gel-Nimbus 26 - Xanh Ngọc",
        "brand": "ASICS",
        "category": "Road Running",
        "description": "ASICS Gel-Nimbus 26 với FF BLAST™ PLUS ECO foam tái chế và hệ thống PureGEL™. Đây là đôi giày long distance hoàn hảo cho những buổi chạy marathon.",
        "price": 3900000,
        "original_price": 4800000,
        "colors": [
            {"name": "Xanh Ngọc", "hex": "#0F766E"},
            {"name": "Tím Nhạt", "hex": "#A78BFA"},
        ],
        "sizes": ["36", "37", "38", "39", "40", "41"],
        "images": [
            "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=600&q=80",
            "https://images.unsplash.com/photo-1587563871167-1ee9c731aefb?w=600&q=80",
        ],
        "tags": ["road", "running", "women", "asics", "gel-nimbus", "marathon", "long-distance"],
        "is_new": False,
        "in_stock": True,
        "stock_qty": 21,
        "rating": 4.8,
        "review_count": 112,
    },
    # NEW BALANCE
    {
        "name": "Giày Chạy Bộ Nam New Balance Fresh Foam X 1080v13 - Xám",
        "brand": "NEW BALANCE",
        "category": "Road Running",
        "description": "New Balance 1080v13 với Fresh Foam X được thiết kế lại giúp tối ưu hóa độ mềm mại và độ bền. Phần upper mới Hypoknit thoáng khí và ôm chân hoàn hảo.",
        "price": 4300000,
        "original_price": None,
        "colors": [
            {"name": "Xám", "hex": "#6B7280"},
            {"name": "Đen Trắng", "hex": "#374151"},
        ],
        "sizes": ["40", "41", "42", "43", "44", "45"],
        "images": [
            "https://images.unsplash.com/photo-1606107557195-0e29a4b5b4aa?w=600&q=80",
            "https://images.unsplash.com/photo-1460353581641-37baddab0fa2?w=600&q=80",
        ],
        "tags": ["road", "running", "men", "new-balance", "fresh-foam", "1080"],
        "is_new": True,
        "in_stock": True,
        "stock_qty": 14,
        "rating": 4.7,
        "review_count": 65,
    },
    {
        "name": "Giày Chạy Bộ Nữ New Balance 860v14 - Hồng Đậm",
        "brand": "NEW BALANCE",
        "category": "Road Running",
        "description": "New Balance 860v14 stability shoe với Fresh Foam X midsole và ROLLBAR® post. Được thiết kế cho người chạy bộ cần hỗ trợ bổ sung mà không hy sinh sự thoải mái.",
        "price": 3100000,
        "original_price": 3800000,
        "colors": [
            {"name": "Hồng Đậm", "hex": "#EC4899"},
            {"name": "Xanh Biển", "hex": "#0EA5E9"},
        ],
        "sizes": ["36", "37", "38", "39", "40"],
        "images": [
            "https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?w=600&q=80",
            "https://images.unsplash.com/photo-1539185441755-769473a23570?w=600&q=80",
        ],
        "tags": ["road", "running", "women", "new-balance", "860", "stability", "rollbar"],
        "is_new": False,
        "in_stock": True,
        "stock_qty": 32,
        "rating": 4.5,
        "review_count": 48,
    },
]


async def seed():
    client = AsyncIOMotorClient(settings.mongodb_url)
    db = client[settings.mongodb_db]

    # Clear existing data
    await db.products.delete_many({})
    print("Cleared existing products")

    # Connect Milvus
    connect_milvus()

    inserted = 0
    for product in PRODUCTS:
        result = await db.products.insert_one(product.copy())
        product_id = str(result.inserted_id)
        text = f"{product['name']} {product['brand']} {product['category']} {product['description']} {' '.join(product['tags'])}"
        try:
            upsert_product_embedding(product_id, text)
        except Exception as e:
            print(f"  Milvus warning for {product['name']}: {e}")
        inserted += 1
        print(f"  Inserted: {product['name']}")

    print(f"\nDone! Seeded {inserted} products.")
    client.close()


if __name__ == "__main__":
    asyncio.run(seed())
