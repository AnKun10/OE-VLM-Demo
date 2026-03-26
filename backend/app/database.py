from motor.motor_asyncio import AsyncIOMotorClient
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from app.config import settings

# MongoDB
mongo_client: AsyncIOMotorClient = None
mongo_db = None


async def connect_mongodb():
    global mongo_client, mongo_db
    mongo_client = AsyncIOMotorClient(settings.mongodb_url)
    mongo_db = mongo_client[settings.mongodb_db]
    # Create indexes
    await mongo_db.products.create_index("brand")
    await mongo_db.products.create_index("category")
    await mongo_db.products.create_index([("name", "text"), ("description", "text"), ("tags", "text")])
    print("Connected to MongoDB")


async def disconnect_mongodb():
    if mongo_client:
        mongo_client.close()
        print("Disconnected from MongoDB")


def get_db():
    return mongo_db


# Milvus
def connect_milvus():
    try:
        connections.connect(
            alias="default",
            host=settings.milvus_host,
            port=settings.milvus_port,
        )
        _ensure_collection()
        print("Connected to Milvus")
    except Exception as e:
        print(f"Warning: Could not connect to Milvus: {e}. Semantic search will be unavailable.")


def disconnect_milvus():
    try:
        connections.disconnect("default")
        print("Disconnected from Milvus")
    except Exception:
        pass


def _ensure_collection():
    collection_name = settings.milvus_collection
    if utility.has_collection(collection_name):
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    ]
    schema = CollectionSchema(fields=fields, description="Product embeddings")
    collection = Collection(name=collection_name, schema=schema)
    collection.create_index(
        field_name="embedding",
        index_params={"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
    )
    print(f"Created Milvus collection: {collection_name}")


def get_milvus_collection() -> Collection | None:
    try:
        col = Collection(settings.milvus_collection)
        col.load()
        return col
    except Exception:
        return None
