from motor.motor_asyncio import AsyncIOMotorClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PayloadSchemaType, VectorParams

from app.config import settings
from app.services.clip_service import get_vector_size

# MongoDB
mongo_client: AsyncIOMotorClient | None = None
mongo_db = None


async def connect_mongodb():
    global mongo_client, mongo_db
    mongo_client = AsyncIOMotorClient(settings.mongodb_url)
    mongo_db = mongo_client[settings.mongodb_db]
    print("Connected to MongoDB")


async def disconnect_mongodb():
    global mongo_client, mongo_db
    if mongo_client is not None:
        mongo_client.close()
        mongo_client = None
        mongo_db = None
        print("Disconnected from MongoDB")


def get_db():
    return mongo_db


# Qdrant (embedded, local file storage)
_qdrant_client: QdrantClient | None = None


def connect_qdrant() -> None:
    global _qdrant_client
    try:
        _qdrant_client = QdrantClient(path=settings.qdrant_path)
    except Exception as e:
        raise RuntimeError(
            f"Qdrant storage '{settings.qdrant_path}' is locked or inaccessible. "
            f"Stop the backend before seeding (or vice versa). Original: {e}"
        ) from e
    _ensure_qdrant_collection()
    print(f"Connected to Qdrant at {settings.qdrant_path}")


def disconnect_qdrant() -> None:
    global _qdrant_client
    if _qdrant_client is not None:
        try:
            _qdrant_client.close()
        finally:
            _qdrant_client = None
            print("Disconnected from Qdrant")


def get_qdrant_client() -> QdrantClient | None:
    return _qdrant_client


def _ensure_qdrant_collection() -> None:
    assert _qdrant_client is not None
    collection_name = settings.qdrant_collection
    existing = {c.name for c in _qdrant_client.get_collections().collections}
    if collection_name in existing:
        return

    vector_size = get_vector_size()
    _qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    _qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="store",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    _qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="category",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print(f"Created Qdrant collection: {collection_name} (dim={vector_size})")
