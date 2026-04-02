from typing import Optional
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.models.product import ProductResponse, ProductListResponse, FilterOptions
from app.services.milvus_service import search_similar_products

DEFAULT_STORES = ["AnStore", "ThanhStore", "TuanAnhStore"]
DEFAULT_LAYERS = ["Background", "Texture", "Art"]


def _serialize(doc: dict) -> ProductResponse:
    doc["id"] = str(doc.pop("_id"))
    return ProductResponse(**doc)


async def _fetch_products_by_ids(
    db: AsyncIOMotorDatabase,
    ids: list[str],
    stores: Optional[list[str]] = None,
    layers: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
) -> list[dict]:
    valid_object_ids = [ObjectId(product_id) for product_id in ids if ObjectId.is_valid(product_id)]
    if not valid_object_ids:
        return []

    mongo_query: dict = {"_id": {"$in": valid_object_ids}}
    if stores:
        mongo_query["store"] = {"$in": stores}
    if layers:
        mongo_query["layer"] = {"$in": layers}
    if categories:
        mongo_query["category"] = {"$in": categories}

    docs = await db.products.find(mongo_query).to_list(length=len(valid_object_ids))
    doc_map = {str(doc["_id"]): doc for doc in docs}
    return [doc_map[product_id] for product_id in ids if product_id in doc_map]

async def get_products(
    db: AsyncIOMotorDatabase,
    page: int = 1,
    page_size: int = 12,
    search: Optional[str] = None,
    stores: Optional[list[str]] = None,
    layers: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
    sort_by: str = "created",
    sort_order: str = "desc",
    semantic_search: bool = False,
) -> ProductListResponse:
    del semantic_search
    query: dict = {}

    if search:
        # TODO: add cache/Redis to save query+ids.
        ids = search_similar_products(
            search,
            top_k=100,
            stores=stores,
            layers=layers,
        )
        skip = (page - 1) * page_size
        ordered_docs = await _fetch_products_by_ids(
            db,
            ids[skip : skip + page_size],
            stores=stores,
            layers=layers,
            categories=categories,
        )
        total = len(ids) # Is top-k: 100
        paged_docs = ordered_docs
        items = [_serialize(doc) for doc in paged_docs]
        return ProductListResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=max(1, (total + page_size - 1) // page_size),
        )

    if stores:
        query["store"] = {"$in": stores}
    if layers:
        query["layer"] = {"$in": layers}
    if categories:
        query["category"] = {"$in": categories}

    sort_field_map = {
        "created": "_id",
        "name": "name",
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
    stores = await db.products.distinct("store")
    categories = await db.products.distinct("category")

    return FilterOptions(
        stores=sorted(set(DEFAULT_STORES) | set(stores)),
        categories=sorted(categories),
        layers=DEFAULT_LAYERS,
    )


async def get_related_products(
    db: AsyncIOMotorDatabase, product_id: str, limit: int = 4
) -> list[ProductResponse]:
    product = await get_product_by_id(db, product_id)
    if not product:
        return []
    ids = search_similar_products(f"{product.name} {product.store} {product.category}", top_k=limit + 1)
    ids = [i for i in ids if i != product_id][:limit]
    if not ids:
        # Fallback: same category
        cursor = (
            db.products.find({"category": product.category, "_id": {"$ne": ObjectId(product_id)}})
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        return [_serialize(doc) for doc in docs]
    docs = await _fetch_products_by_ids(db, ids)
    return [_serialize(doc) for doc in docs]
