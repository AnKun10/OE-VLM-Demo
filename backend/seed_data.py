"""
Seed script: populates MongoDB with sample fashion products and indexes them in Milvus.
Run from the backend/ directory:
    python seed_data.py
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
from app.database import connect_milvus
from app.services.milvus_service import upsert_product_embedding

PRODUCTS = []

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
        text = f"{product['name']} {product['store']} {product['category']} {product['description']}"
        try:
            upsert_product_embedding(
                product_id,
                text,
                store=product["store"],
            )
        except Exception as e:
            print(f"  Milvus warning for {product['name']}: {e}")
        inserted += 1
        print(f"  Inserted: {product['name']}")

    print(f"\nDone! Seeded {inserted} products.")
    client.close()


if __name__ == "__main__":
    asyncio.run(seed())
