from app.database import get_milvus_collection
from app.services.clip_service import embed_text

def _escape_milvus_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def build_scalar_filter(
    stores: list[str] | None = None,
    layers: list[str] | None = None,
) -> str | None:
    clauses: list[str] = []
    if stores:
        quoted = ", ".join(f'"{_escape_milvus_value(store)}"' for store in stores)
        clauses.append(f"store in [{quoted}]")
    if layers:
        quoted = ", ".join(f'"{_escape_milvus_value(layer)}"' for layer in layers)
        clauses.append(f"layer in [{quoted}]")
    return " and ".join(clauses) if clauses else None


def upsert_product_embedding(
    product_id: str,
    text: str,
    store: str | None = None,
    layer: str | None = None,
):
    col = get_milvus_collection()
    if col is None:
        return
    embedding = embed_text(text).tolist()
    store_value = store or ""
    layer_value = layer or ""
    col.upsert([[product_id], [store_value], [layer_value], [embedding]])
    col.flush()


def search_similar_products(
    query: str,
    top_k: int = 20,
    stores: list[str] | None = None,
    layers: list[str] | None = None,
) -> list[str]:
    col = get_milvus_collection()
    if col is None:
        return []
    query_embedding = embed_text(query).tolist()
    search_kwargs = {
        "data": [query_embedding],
        "anns_field": "vector",
        "limit": top_k,
        "output_fields": ["store", "layer"],
        "param": {"metric_type": "COSINE", "params": {"nprobe": 16}},
    }
    expr = build_scalar_filter(stores=stores, layers=layers)
    if expr:
        search_kwargs["expr"] = expr

    try:
        results = col.search(**search_kwargs)
    except Exception:
        if expr:
            search_kwargs.pop("expr", None)
            results = col.search(**search_kwargs)
        else:
            raise

    if not results or len(results[0]) == 0:
        return []
    return [hit.id for hit in results[0]]
