from typing import Optional
from sentence_transformers import SentenceTransformer
from app.database import get_milvus_collection
from app.config import settings

_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_text(text: str) -> list[float]:
    model = get_embedding_model()
    return model.encode(text).tolist()


def upsert_product_embedding(product_id: str, text: str):
    col = get_milvus_collection()
    if col is None:
        return
    embedding = embed_text(text)
    col.upsert([[product_id], [embedding]])
    col.flush()


def search_similar_products(query: str, top_k: int = 20) -> list[str]:
    col = get_milvus_collection()
    if col is None:
        return []
    query_embedding = embed_text(query)
    results = col.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=top_k,
        output_fields=["id"],
    )
    if not results or len(results[0]) == 0:
        return []
    return [hit.entity.get("id") for hit in results[0]]
