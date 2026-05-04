# FG-CLIP 2 + Qdrant Migration — Design

**Date**: 2026-04-24
**Branch**: `dev/anhnt2112_MetaCLIP` (or new `dev/anhnt2112_FGCLIP2`)
**Reference**: `/home/anhnt2112/Documents/OE_Embedding/method/fg_clip_2_transparent_bg_early_fusion.ipynb`

## Goal

Replace the embedding model (MetaCLIP 2 → FG-CLIP 2 base) and vector database (Milvus → Qdrant local file storage) in the OE-VLM-Demo backend. Index products with an image-based early-fusion embedding (the notebook pipeline); keep the runtime text-query path unchanged. Seed data from a CSV file via a CLI script.

## Scope decisions (locked during brainstorming)

1. **Index vector = image early-fusion.** `fused = normalize(0.9 * image_embedding + 0.1 * text_embedding("transparent background, isolated object"))`. Image is preprocessed with the notebook's "method 2": if `transparent_ratio > 0.05`, composite the RGBA image onto a `(127,127,127)` gray background; otherwise convert RGB.
2. **Query path unchanged.** Text queries call `embed_text()` and Qdrant cosine search — semantics equivalent to the current Milvus flow.
3. **Vector DB = Qdrant local file storage** via `QdrantClient(path=...)`. **Not Docker.** Single-process file lock: backend and seed script cannot run concurrently.
4. **CSV schema = TIB notebook format**, mapped to the OE-VLM product schema:
   - `clipart_name` → `name`
   - `url` → `image_url`
   - `clipart_category_name` → `category`
   - `store` = `"Sample User"` (hardcoded)
   - `layer` and `description` fields are removed from the product schema entirely.
5. **Frontend updated in-sync** to drop Layer filter UI and Description rendering.
6. **Seed CLI**: `python seed_data.py --csv <path>` with a default fallback path.
7. **Failed image downloads/embeddings**: skip the row entirely (no Mongo insert, no Qdrant insert), log to a JSON failures file.

## High-level architecture

```
CSV file ─┐
          ▼
   seed_data.py (CLI: --csv)
          │
          ├─► fetch image (URL) → detect method 2 → apply method 2 → FG-CLIP2 image embed
          │                                                             │
          │                                                             ▼
          │                                    early-fusion: 0.9·img + 0.1·text_prompt
          │                                                             │  (L2-normalize)
          │                                                             ▼
          ├─► Mongo: insert product doc (name, image_url, store="Sample User", category)
          └─► Qdrant: upsert(id=product_id, vector=fused_768d, payload={store, category})

Query flow (semantics unchanged):
   user text → FG-CLIP2 get_text_features(walk_type="long") → L2-normalize
             → Qdrant cosine search (top_k) → list of product_ids
             → Mongo fetch docs
```

## Backend components

### `config.py`

```python
class Settings(BaseSettings):
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db: str = "fashion_db"
    qdrant_path: str = "./qdrant_storage"     # relative to backend/ CWD
    qdrant_collection: str = "products"
    fgclip_model_id: str = "qihoo360/fg-clip2-base"
    fusion_text: str = "transparent background, isolated object"
    fusion_weight_image: float = 0.9
    fusion_weight_text: float = 0.1

    class Config:
        env_file = ".env"
```

Removed: `metaclip_model_id`, `milvus_host`, `milvus_port`, `milvus_collection`.

### `services/clip_service.py` (rewrite)

Public API kept compatible so existing callers (`milvus_service` → `qdrant_service`, `product_service`) only change vector-DB names:

- `load_clip_model()`, `unload_clip_model()`
- `get_vector_size() -> int` returns `768`
- `get_runtime_device() -> str`
- `embed_text(text: str) -> np.ndarray` — uses `tokenizer(..., padding="max_length", truncation=True, max_length=196)` and `model.get_text_features(**tokens, walk_type="long")`, then L2-normalize.
- `embed_image(pil_rgba: Image.Image) -> np.ndarray` — new logic:
  1. `detect_method_2(rgba)` → check alpha channel, compute `transparent_ratio = (~fg_mask).sum() / area`; return True if `> 0.05` and `fg_count >= 20`
  2. If method 2 → `apply_method_2(pil_rgba, bg=(127,127,127))` (alpha composite)
  3. Else → `.convert("RGB")`
  4. `image_processor(images=rgb, max_num_patches=determine_max_patches(rgb), return_tensors="pt")`
  5. `model.get_image_features(**inputs)`, L2-normalize
- `early_fusion_embed(pil_rgba: Image.Image) -> np.ndarray` — new helper:
  - Returns `l2_normalize(W_IMAGE * embed_image(pil) + W_TEXT * _fusion_text_vec)`
  - `_fusion_text_vec` is a module-level cache populated in `load_clip_model()` by embedding `settings.fusion_text` once.

Helpers (module-private):
- `detect_method_2`, `apply_method_2`, `determine_max_patches`, `l2_normalize` — copy from notebook verbatim.

Internals use `AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)`, `AutoTokenizer`, `AutoImageProcessor`. CUDA-first with CPU fallback on `RuntimeError`.

### `services/qdrant_service.py` (new, replaces `milvus_service.py`)

Public API parallels the removed Milvus service so consumers change imports only:

```python
def upsert_product_embedding(
    product_id: str,
    embedding: np.ndarray | list[float],
    store: str,
    category: str,
) -> None: ...

def search_similar_products(
    query: str,
    top_k: int = 20,
    stores: list[str] | None = None,
) -> list[str]: ...
```

**Signature change vs. current Milvus service**:
- `upsert` now takes `embedding` directly (seed computes it from image) instead of `text`. This is the one breaking change in the service API, and `seed_data.py` is the only caller of `upsert`.
- `search_similar_products` drops the `layers` parameter. Categories continue to be filtered at the Mongo layer (matching current behavior), not at the vector DB.

Query flow inside `search_similar_products`:
1. `vec = embed_text(query)`
2. Build Qdrant `Filter` from `stores` using `FieldCondition(key="store", match=MatchAny(any=stores))`. No filter applied when `stores` is falsy.
3. `client.search(collection_name=..., query_vector=vec.tolist(), limit=top_k, query_filter=...)`
4. Return `[hit.id for hit in result]`.

Uses `qdrant_client.QdrantClient` singleton from `database.py`. Distance `Cosine`.

### `database.py`

Remove Milvus code. Add:

```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PayloadSchemaType

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

def disconnect_qdrant() -> None:
    global _qdrant_client
    if _qdrant_client is not None:
        _qdrant_client.close()
        _qdrant_client = None

def get_qdrant_client() -> QdrantClient | None:
    return _qdrant_client

def _ensure_qdrant_collection() -> None:
    # If the collection is absent, create it with 768-dim cosine vectors
    # and payload indexes on `store` (keyword) so Qdrant filters stay fast.
    # No-op if the collection already exists.
    ...
```

(The `...` bodies are design-level intent; the exact Qdrant client calls are left to the implementation plan — e.g. `client.get_collections()`, `client.create_collection(..., vectors_config=VectorParams(size=768, distance=Distance.COSINE))`, `client.create_payload_index(..., field_name="store", field_schema=PayloadSchemaType.KEYWORD)`.)

Error handling: if `QdrantClient(path=...)` raises a lock error (another process holds the storage), wrap and re-raise with a clear message: `"Qdrant storage '<path>' is locked. Stop the backend before seeding (or vice versa)."`

### `models/product.py`

Remove `layer` and `description` fields from `ProductCreate`, `ProductResponse`. Remove `layers: list[str]` from `FilterOptions`.

```python
class ProductResponse(BaseModel):
    id: str
    name: str
    image_url: str
    store: str
    category: str
```

### `services/product_service.py`

- Remove `layers` parameter from `get_products`, `_fetch_products_by_ids`.
- Remove `DEFAULT_LAYERS`.
- `get_filter_options` returns only `stores` and `categories`.
- `get_related_products` query text stays `f"{product.name} {product.store} {product.category}"`.
- The `semantic_search` parameter is already a no-op (`del semantic_search`); leave as-is — not in scope to clean up.

### `routers/products.py`

- Drop the `layers: Optional[list[str]]` query parameter from `list_products`.
- No route removals.

### `main.py`

- Replace `connect_milvus / disconnect_milvus` with `connect_qdrant / disconnect_qdrant`.
- Lifespan order unchanged: Mongo → model → vector DB on startup; reverse on shutdown.

### `seed_data.py` (rewrite)

```
python seed_data.py --csv path/to/clipart.csv [--failures-out failures.json]
```

Flow:
1. `argparse` with `--csv` (default `./data/products.csv`) and `--failures-out` (default `./failures.json`).
2. Load CSV with pandas. Normalize each row to `{clipart_id, url, clipart_name, clipart_category_name}`. Drop rows with missing/invalid `url` or `clipart_id`.
3. Boot: `load_clip_model()`, `connect_qdrant()`, Mongo client.
4. Clear existing data: `db.products.delete_many({})`, drop+recreate Qdrant collection.
5. Iterate rows with `tqdm`:
   - `fetch_image(url)` → PIL RGBA (timeout 30s)
   - `fused = early_fusion_embed(rgba)`
   - Build Mongo doc `{name, image_url, store: "Sample User", category}`, `insert_one` to get `_id`
   - `upsert_product_embedding(str(_id), fused, store, category)`
   - On any exception → append `{clipart_id, url, error: str(e)}` to `errors`, continue.
6. Dump `failures.json` if `errors` non-empty. Print summary `"Seeded N, failed M"`.

## Frontend changes

Concrete touch points surveyed from the repo:

- `frontend/src/types/index.ts:6` — remove `layer: string` from `Product`
- `frontend/src/types/index.ts:8` — remove `description: string` from `Product`
- `frontend/src/types/index.ts:22` — remove `layers: string[]` from `FilterOptions`
- `frontend/src/lib/api.ts:10` — remove `layers?: string[]` from query params type
- `frontend/src/components/ProductCard.tsx:74` — remove `{product.layer}` render
- `frontend/src/pages/ProductDetailPage.tsx:108` — remove `{product.description}` paragraph
- `frontend/src/pages/ProductListPage.tsx` — remove layer-related code: `selectedLayers` state, `toggleLayer`, Layer filter section (lines ~154–168), layer chip badges (~278–281), `layers` in query params, active-filter count.

## Dependencies

`backend/requirements.txt`:

- Remove `pymilvus==2.4.9`.
- Add `qdrant-client>=1.11.0`.
- Add `pandas>=2.0` (CSV reading).
- Add `tqdm>=4.66` (seed progress).
- Add `requests>=2.32` (fetch images).
- Bump `transformers==4.46.0` → `transformers>=4.56.0` (FG-CLIP 2 compatibility).
- Keep `torch==2.5.1` initially. If the FG-CLIP 2 remote code requires a newer torch, upgrade during implementation — do not upgrade preemptively.
- Keep `pillow==11.0.0`, `numpy==1.26.4`.

## `docker-compose.yml`

- Remove `milvus`, `etcd`, `minio` services and their volumes.
- Keep `mongodb` service.
- **Do not add** Qdrant — it runs embedded in the backend process.

## `.env.example`

```
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB=fashion_db
QDRANT_PATH=./qdrant_storage
QDRANT_COLLECTION=products
FGCLIP_MODEL_ID=qihoo360/fg-clip2-base
```

## `.gitignore`

Add `backend/qdrant_storage/`.

## `CLAUDE.md`

Update Architecture section:
- `MongoDB 7 via Motor` → unchanged.
- `Milvus v2.4.13 with 384-dim cosine embeddings (all-MiniLM-L6-v2)` → `Qdrant local file storage (cosine, 768-dim HNSW) using qihoo360/fg-clip2-base via AutoModelForCausalLM (trust_remote_code=True)`.
- Update the ports list: drop 19530; do not add a Qdrant port.
- Update the Embedding pipeline note: products are embedded via image early-fusion at seed time; text queries use get_text_features with walk_type="long".
- Update the `docker compose up -d` description: now starts only MongoDB.
- Seed command stays `python seed_data.py --csv <path>`.

## Error handling summary

| Failure | Response |
| --- | --- |
| Model load fails on CUDA | Fallback to CPU (existing pattern) |
| `trust_remote_code` unavailable | Fail fast at startup |
| CSV file missing | Raise `FileNotFoundError`, exit 1 |
| Row missing `url` / `clipart_id` | Skip row, log to failures |
| Image fetch error (HTTP/DNS/timeout) | Skip row, log to failures |
| PIL cannot decode image | Skip row, log to failures |
| Image embedding exception | Skip row, log to failures |
| Mongo/Qdrant insert error (infra) | Raise and halt seed |
| Qdrant storage file-locked | Raise with clear message pointing to dual-process cause |
| Runtime query: Qdrant client None | `search_similar_products` returns `[]` |
| Runtime query: `embed_text` fails | Log and return `[]` |

## Testing strategy

Repo has no test framework. Manual smoke verification only:

1. **Seed smoke**: `python seed_data.py --csv <small-csv>` → verify `qdrant_storage/` created, Mongo populated, no lock error, `failures.json` present if any row failed.
2. **API smoke**: `uvicorn app.main:app` + `curl` for `/api/products`, `/api/products?search=<query>`, `/api/products/{id}/related`, `/api/products/filters`. Verify responses have no `layer`/`description`/`layers`.
3. **Frontend smoke**: `npm run dev` → no Layer filter, no description section, search returns semantic results.
4. **Failure smoke**: CSV with one unreachable URL → confirm skip behavior and failures JSON.

No pytest/vitest added — out of scope for this demo migration.

## Rollback plan

- Implementation happens on a dedicated branch.
- Previous Milvus/MetaCLIP state is preserved at commit `fc91981`. `git revert` the migration commits if rollback is required.

## Known limitations / open questions (to verify at implementation)

1. **Torch version**: notebook used `torch 2.11.0+cu130`. Keep `torch==2.5.1`; upgrade only if FG-CLIP 2 remote code fails on it.
2. **transformers 4.46 → 4.56+** is a significant jump; re-run smoke tests after upgrade.
3. **Seed duration**: notebook took ~44 min for 4161 rows on a laptop GPU. Document in README that seed is a long-running operation.
4. **Query language**: FG-CLIP 2 supports EN/ZH only (per `Note.md`), while the demo UI is Vietnamese. Vietnamese text queries will produce poor semantic recall. **Noted as known limitation**; not addressed in this migration (a future translation layer could solve it).
5. **HNSW params**: use Qdrant defaults (`m=16`, `ef_construct=100`, search `ef=128`). No tuning in scope.
