from __future__ import annotations

import uuid

import numpy as np
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
)

from app.config import settings
from app.database import get_qdrant_client
from app.services.clip_service import embed_text


def _point_id_for(product_id: str) -> str:
    # Qdrant requires UUID or unsigned int IDs. We derive a stable UUIDv5 from the
    # Mongo ObjectId string so upserts are idempotent across runs.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"product:{product_id}"))


def upsert_product_embedding(
    product_id: str,
    embedding: np.ndarray | list[float],
    store: str,
    category: str,
) -> None:
    client = get_qdrant_client()
    if client is None:
        return
    vector = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
    point = PointStruct(
        id=_point_id_for(product_id),
        vector=vector,
        payload={
            "product_id": product_id,
            "store": store or "",
            "category": category or "",
        },
    )
    client.upsert(collection_name=settings.qdrant_collection, points=[point])


def _build_filter(stores: list[str] | None) -> Filter | None:
    if not stores:
        return None
    return Filter(
        must=[FieldCondition(key="store", match=MatchAny(any=list(stores)))]
    )


def search_similar_products(
    query: str,
    top_k: int = 20,
    stores: list[str] | None = None,
) -> list[str]:
    client = get_qdrant_client()
    if client is None:
        return []
    try:
        vec = embed_text(query).tolist()
    except Exception:
        return []

    try:
        hits = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=vec,
            limit=top_k,
            query_filter=_build_filter(stores),
            with_payload=True,
        )
    except Exception:
        return []

    return [str(hit.payload.get("product_id", "")) for hit in hits if hit.payload]
