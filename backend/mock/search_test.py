"""
Direct Qdrant search test — bypasses Mongo, filters, and the FastAPI layer.
Confirms the embedding+vector-search path works end to end.

Run from backend/ with backend (uvicorn) STOPPED:
    python mock/search_test.py "a red running shoe"
    python mock/search_test.py        # uses default query
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient

from app.config import settings
from app.services.clip_service import embed_text, load_clip_model


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "a red running shoe"
    print(f"Query: {query!r}")

    print("Loading FG-CLIP 2 (first run downloads model, may take minutes)...")
    load_clip_model()
    vec = embed_text(query)
    print(f"Vector dim: {vec.shape[0]}")

    try:
        c = QdrantClient(path=settings.qdrant_path)
    except Exception as e:
        print(f"Cannot open qdrant_storage — backend still running? Error: {e}")
        return

    try:
        # Search WITHOUT any filter to verify raw retrieval works.
        res = c.query_points(
            collection_name=settings.qdrant_collection,
            query=vec.tolist(),
            limit=5,
            with_payload=True,
        )
        hits = res.points
        print(f"\nNo-filter hits: {len(hits)}")
        for h in hits:
            print(f"  score={h.score:.4f}  payload={h.payload}")

        # Search WITH the same filter the API uses (store='Sample User').
        from qdrant_client.http.models import FieldCondition, Filter, MatchAny

        filt = Filter(must=[FieldCondition(key="store", match=MatchAny(any=["Sample User"]))])
        res2 = c.query_points(
            collection_name=settings.qdrant_collection,
            query=vec.tolist(),
            limit=5,
            query_filter=filt,
            with_payload=True,
        )
        hits2 = res2.points
        print(f"\nFiltered hits (store='Sample User'): {len(hits2)}")
        for h in hits2:
            print(f"  score={h.score:.4f}  payload={h.payload}")

        if hits and not hits2:
            print("\n>>> Filter is excluding everything — payload.store ≠ 'Sample User'.")
            print("    Check what stores actually exist in Qdrant payloads (see diagnose.py).")
    finally:
        c.close()


if __name__ == "__main__":
    main()
