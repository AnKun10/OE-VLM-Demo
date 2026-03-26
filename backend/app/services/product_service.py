from typing import Optional
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.models.product import ProductCreate, ProductResponse, ProductListResponse, FilterOptions
from app.services.milvus_service import upsert_product_embedding, search_similar_products


def _serialize(doc: dict) -> ProductResponse:
    doc["id"] = str(doc.pop("_id"))
    # Compute discount percent
    price = doc.get("price", 0)
    orig = doc.get("original_price")
    if orig and orig > price:
        doc["discount_percent"] = round((1 - price / orig) * 100)
    else:
        doc["discount_percent"] = None
    return ProductResponse(**doc)


async def create_product(db: AsyncIOMotorDatabase, product: ProductCreate) -> ProductResponse:
    doc = product.model_dump()
    result = await db.products.insert_one(doc)
    created = await db.products.find_one({"_id": result.inserted_id})
    response = _serialize(created)
    # Index in Milvus
    text = f"{product.name} {product.brand} {product.category} {product.description} {' '.join(product.tags)}"
    upsert_product_embedding(response.id, text)
    return response


async def get_products(
    db: AsyncIOMotorDatabase,
    page: int = 1,
    page_size: int = 12,
    search: Optional[str] = None,
    brands: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    sort_by: str = "created",
    sort_order: str = "desc",
    semantic_search: bool = False,
) -> ProductListResponse:
    query: dict = {}

    # Semantic search via Milvus
    if search and semantic_search:
        ids = search_similar_products(search, top_k=100)
        if ids:
            query["_id"] = {"$in": [ObjectId(i) for i in ids if ObjectId.is_valid(i)]}
    elif search:
        query["$text"] = {"$search": search}

    if brands:
        query["brand"] = {"$in": brands}
    if categories:
        query["category"] = {"$in": categories}
    if min_price is not None or max_price is not None:
        price_filter: dict = {}
        if min_price is not None:
            price_filter["$gte"] = min_price
        if max_price is not None:
            price_filter["$lte"] = max_price
        query["price"] = price_filter

    sort_field_map = {
        "created": "_id",
        "price": "price",
        "name": "name",
        "rating": "rating",
        "discount": "original_price",
    }
    mongo_sort_field = sort_field_map.get(sort_by, "_id")
    mongo_sort_dir = -1 if sort_order == "desc" else 1

    total = await db.products.count_documents(query)
    skip = (page - 1) * page_size

    cursor = db.products.find(query).sort(mongo_sort_field, mongo_sort_dir).skip(skip).limit(page_size)
    docs = await cursor.to_list(length=page_size)
    items = [_serialize(doc) for doc in docs]

    return ProductListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=max(1, (total + page_size - 1) // page_size),
    )


async def get_product_by_id(db: AsyncIOMotorDatabase, product_id: str) -> Optional[ProductResponse]:
    if not ObjectId.is_valid(product_id):
        return None
    doc = await db.products.find_one({"_id": ObjectId(product_id)})
    if not doc:
        return None
    return _serialize(doc)


async def get_filter_options(db: AsyncIOMotorDatabase) -> FilterOptions:
    brands = await db.products.distinct("brand")
    categories = await db.products.distinct("category")
    pipeline = [
        {"$group": {"_id": None, "min_price": {"$min": "$price"}, "max_price": {"$max": "$price"}}}
    ]
    price_result = await db.products.aggregate(pipeline).to_list(1)
    min_price = price_result[0]["min_price"] if price_result else 0
    max_price = price_result[0]["max_price"] if price_result else 0

    return FilterOptions(
        brands=sorted(brands),
        categories=sorted(categories),
        min_price=min_price,
        max_price=max_price,
    )


async def get_related_products(
    db: AsyncIOMotorDatabase, product_id: str, limit: int = 4
) -> list[ProductResponse]:
    product = await get_product_by_id(db, product_id)
    if not product:
        return []
    ids = search_similar_products(f"{product.name} {product.brand} {product.category}", top_k=limit + 1)
    ids = [i for i in ids if i != product_id][:limit]
    if not ids:
        # Fallback: same category
        cursor = (
            db.products.find({"category": product.category, "_id": {"$ne": ObjectId(product_id)}})
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        return [_serialize(doc) for doc in docs]
    docs = await db.products.find({"_id": {"$in": [ObjectId(i) for i in ids if ObjectId.is_valid(i)]}}).to_list(limit)
    return [_serialize(doc) for doc in docs]
