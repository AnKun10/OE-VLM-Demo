"""
Diagnostic script to debug 'search returns 0 products'.

Run from backend/ with backend (uvicorn) STOPPED:
    python mock/diagnose.py

Reports settings, Mongo state, Qdrant state, and a sample payload.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymongo import MongoClient
from qdrant_client import QdrantClient

from app.config import settings


def main() -> None:
    print("=== Settings backend đang dùng ===")
    print(f"mongodb_url:  {settings.mongodb_url}")
    print(f"mongodb_db:   {settings.mongodb_db}")
    print(f"qdrant_path:  {settings.qdrant_path}")
    print(f"qdrant_coll:  {settings.qdrant_collection}")
    print(f"cwd:          {Path.cwd()}")
    print(f"resolved qdrant: {Path(settings.qdrant_path).resolve()}")
    print()

    print("=== Mongo ===")
    try:
        m = MongoClient(settings.mongodb_url, serverSelectionTimeoutMS=2000)[settings.mongodb_db]
        count = m.products.count_documents({})
        print(f"product count :  {count}")
        print(f"distinct store:  {m.products.distinct('store')}")
        sample = m.products.find_one()
        if sample:
            sample.pop("_id", None)
            print(f"sample doc    :  {sample}")
        else:
            print("sample doc    :  (none)")
    except Exception as e:
        print(f"  Mongo error: {type(e).__name__}: {e}")
    print()

    print("=== Qdrant ===")
    try:
        c = QdrantClient(path=settings.qdrant_path)
    except Exception as e:
        print(f"  Cannot open qdrant_storage. Backend probably still running. Error: {e}")
        return

    try:
        names = [col.name for col in c.get_collections().collections]
        print(f"collections   :  {names}")
        if settings.qdrant_collection not in names:
            print(f"  Collection '{settings.qdrant_collection}' MISSING — seed didn't create it.")
            return
        cnt = c.count(collection_name=settings.qdrant_collection).count
        print(f"point count   :  {cnt}")
        pts, _ = c.scroll(
            collection_name=settings.qdrant_collection,
            limit=2,
            with_payload=True,
            with_vectors=False,
        )
        for i, p in enumerate(pts, 1):
            print(f"sample {i} id  :  {p.id}")
            print(f"sample {i} pl  :  {p.payload}")
    except Exception as e:
        print(f"  Qdrant error: {type(e).__name__}: {e}")
    finally:
        c.close()


if __name__ == "__main__":
    main()
