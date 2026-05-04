"""
Wipe BOTH Mongo `products` collection AND Qdrant `products` collection.
Use when stale data from a previous seed (different schema/model) is poisoning search.

After running this, re-seed:
    python seed_data.py --csv ./data/products.csv

Run from backend/ with backend (uvicorn) STOPPED:
    python mock/reset_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymongo import MongoClient
from qdrant_client import QdrantClient

from app.config import settings


def main() -> None:
    print(f"This will wipe:")
    print(f"  Mongo  {settings.mongodb_url} db={settings.mongodb_db} collection=products")
    print(f"  Qdrant {settings.qdrant_path}  collection={settings.qdrant_collection}")
    confirm = input("Type YES to proceed: ").strip()
    if confirm != "YES":
        print("Aborted.")
        return

    m = MongoClient(settings.mongodb_url)[settings.mongodb_db]
    res = m.products.delete_many({})
    print(f"Mongo: deleted {res.deleted_count} products")

    try:
        c = QdrantClient(path=settings.qdrant_path)
    except Exception as e:
        print(f"Qdrant open failed (backend still running?): {e}")
        return

    try:
        existing = {col.name for col in c.get_collections().collections}
        if settings.qdrant_collection in existing:
            c.delete_collection(collection_name=settings.qdrant_collection)
            print(f"Qdrant: dropped collection '{settings.qdrant_collection}'")
        else:
            print(f"Qdrant: collection '{settings.qdrant_collection}' didn't exist (nothing to drop)")
    finally:
        c.close()

    print("Done. Now run: python seed_data.py --csv ./data/products.csv")


if __name__ == "__main__":
    main()
