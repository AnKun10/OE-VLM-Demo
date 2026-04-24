# FG-CLIP 2 + Qdrant Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace MetaCLIP 2 embeddings and Milvus vector DB with FG-CLIP 2 base image early-fusion embeddings stored in a local (embedded) Qdrant, and seed products from a CSV file.

**Architecture:** At seed time, each product's image is fetched, preprocessed with the notebook's "method 2" alpha-compositing, embedded with FG-CLIP 2 `get_image_features`, and fused with a fixed-prompt text embedding (`0.9·img + 0.1·text`). The fused vector is stored in an embedded Qdrant collection. At query time, user text goes through FG-CLIP 2 `get_text_features(walk_type="long")` and a cosine search in Qdrant returns product IDs, which are resolved in MongoDB. Product schema drops `layer` and `description`; frontend is updated in sync.

**Tech Stack:** FastAPI, MongoDB/Motor, Qdrant (embedded, local file storage), `qdrant-client`, `transformers>=4.56`, `qihoo360/fg-clip2-base` via `AutoModelForCausalLM(trust_remote_code=True)`, React 18/Vite/TypeScript frontend.

**Testing note:** The spec explicitly opts out of adding pytest/vitest (YAGNI for this demo). Each task uses a file-edit → import/smoke-check → commit pattern. End-to-end manual smoke is a dedicated task at the end.

**Reference spec:** `docs/superpowers/specs/2026-04-24-fgclip2-qdrant-migration-design.md`
**Reference notebook:** `/home/anhnt2112/Documents/OE_Embedding/method/fg_clip_2_transparent_bg_early_fusion.ipynb`

---

## File map

**Backend — created:**
- `backend/app/services/qdrant_service.py`

**Backend — deleted:**
- `backend/app/services/milvus_service.py`

**Backend — modified:**
- `docker-compose.yml` — remove milvus/etcd/minio/attu
- `backend/requirements.txt` — swap deps
- `backend/.env.example` — swap env vars
- `.gitignore` — add `backend/qdrant_storage/`
- `backend/app/config.py` — replace settings
- `backend/app/services/clip_service.py` — rewrite for FG-CLIP 2 + early fusion
- `backend/app/database.py` — replace Milvus with Qdrant
- `backend/app/models/product.py` — drop `layer`, `description`
- `backend/app/services/product_service.py` — drop `layers` arg + default layers
- `backend/app/routers/products.py` — drop `layers` query param
- `backend/app/routers/chat.py` — swap import
- `backend/app/main.py` — swap lifespan calls
- `backend/seed_data.py` — rewrite for CSV + early-fusion seeding

**Frontend — modified:**
- `frontend/src/types/index.ts` — drop `layer`, `description`, `layers`
- `frontend/src/lib/api.ts` — drop `layers` in params type
- `frontend/src/components/ProductCard.tsx` — drop `{product.layer}` badge
- `frontend/src/pages/ProductDetailPage.tsx` — drop description section
- `frontend/src/pages/ProductListPage.tsx` — drop Layer filter UI, selectedLayers state, toggleLayer

**Docs — modified:**
- `CLAUDE.md` — update Architecture/Infrastructure sections

---

## Task 1: Update docker-compose.yml — remove Milvus stack

**Files:**
- Modify: `docker-compose.yml`

- [ ] **Step 1.1: Replace file contents**

Write `docker-compose.yml` with:

```yaml
version: "3.8"

services:
  mongodb:
    image: mongo:7
    container_name: oe_vlm_mongo
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db
    restart: unless-stopped

volumes:
  mongo_data:
```

- [ ] **Step 1.2: Verify YAML parses**

Run: `docker compose -f docker-compose.yml config > /dev/null && echo OK`
Expected: `OK`

- [ ] **Step 1.3: Commit**

```bash
git add docker-compose.yml
git commit -m "chore: remove Milvus/etcd/MinIO/Attu from docker-compose

Qdrant will run embedded in the backend process; only MongoDB is needed
in docker now."
```

---

## Task 2: Update backend dependencies and env/gitignore

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/.env.example`
- Modify: `.gitignore`

- [ ] **Step 2.1: Replace `backend/requirements.txt`**

```txt
fastapi==0.115.0
uvicorn[standard]==0.32.0
motor==3.6.0
qdrant-client>=1.11.0
marshmallow>=3.13.0,<4
torch==2.5.1
transformers>=4.56.0
sentencepiece==0.2.0
setuptools<81
pydantic==2.9.2
pydantic-settings==2.6.1
python-dotenv==1.0.1
python-multipart==0.0.12
pillow==11.0.0
numpy==1.26.4
pandas>=2.0
tqdm>=4.66
requests>=2.32
```

- [ ] **Step 2.2: Replace `backend/.env.example`**

```
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB=oe_vlm_shop
QDRANT_PATH=./qdrant_storage
QDRANT_COLLECTION=products
FGCLIP_MODEL_ID=qihoo360/fg-clip2-base
```

- [ ] **Step 2.3: Update `.gitignore`**

Edit `.gitignore`: replace the block

```
# Milvus / MinIO local data
milvus_data/
minio_data/
etcd_data/
```

with:

```
# Qdrant local storage
backend/qdrant_storage/
```

- [ ] **Step 2.4: Install new deps (if developer has venv)**

Run: `cd backend && source venv/bin/activate && pip install -r requirements.txt`
Expected: installs without error (may take a few minutes). If `torch==2.5.1` or `transformers>=4.56.0` fails to resolve, upgrade pip first (`pip install -U pip`) and retry.

- [ ] **Step 2.5: Commit**

```bash
git add backend/requirements.txt backend/.env.example .gitignore
git commit -m "chore: swap pymilvus for qdrant-client; add pandas/tqdm/requests

- Bump transformers to 4.56+ for FG-CLIP 2 compatibility
- Add seed-time deps: pandas, tqdm, requests
- Update .env.example to Qdrant-local-path config
- Ignore backend/qdrant_storage/"
```

---

## Task 3: Replace backend config.py with Qdrant + FG-CLIP 2 settings

**Files:**
- Modify: `backend/app/config.py`

- [ ] **Step 3.1: Replace file contents**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db: str = "oe_vlm_shop"
    qdrant_path: str = "./qdrant_storage"
    qdrant_collection: str = "products"
    fgclip_model_id: str = "qihoo360/fg-clip2-base"
    fusion_text: str = "transparent background, isolated object"
    fusion_weight_image: float = 0.9
    fusion_weight_text: float = 0.1

    class Config:
        env_file = ".env"


settings = Settings()
```

- [ ] **Step 3.2: Smoke-check config imports**

Run: `cd backend && python -c "from app.config import settings; print(settings.fgclip_model_id, settings.qdrant_path)"`
Expected: prints `qihoo360/fg-clip2-base ./qdrant_storage` (or the values from your `.env` if set).

Note: this will break other imports (`clip_service`, `database`) until Tasks 4–5 land. That's expected; we commit at the end of this task anyway because each task changes one file atomically.

- [ ] **Step 3.3: Commit**

```bash
git add backend/app/config.py
git commit -m "refactor(config): replace MetaCLIP/Milvus settings with FG-CLIP 2 + Qdrant

Knows nothing about Milvus anymore. Adds fusion prompt + weights as
tunable settings so seeding can be adjusted without code changes."
```

---

## Task 4: Rewrite clip_service.py for FG-CLIP 2 with early fusion

**Files:**
- Modify: `backend/app/services/clip_service.py`

- [ ] **Step 4.1: Replace file contents**

```python
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

from app.config import settings

MODEL_ID = settings.fgclip_model_id
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model: Any | None = None
_tokenizer: Any | None = None
_image_processor: Any | None = None
_fusion_text_vec: np.ndarray | None = None
_vector_size: int | None = None
_runtime_device: str | None = None


def _move_inputs_to_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(x))
    if denom <= 1e-12:
        return x
    return (x / denom).astype(np.float32)


def _detect_method_2(pil_rgba: Image.Image) -> bool:
    rgba = np.array(pil_rgba.convert("RGBA"))
    alpha = rgba[:, :, 3].astype(np.float32)
    h, w = alpha.shape
    fg_mask = alpha > 5
    fg_count = int(fg_mask.sum())
    if fg_count < 20:
        return False
    transparent_ratio = float((~fg_mask).sum()) / max(h * w, 1)
    return transparent_ratio > 0.05


def _apply_method_2(pil_rgba: Image.Image, bg: tuple[int, int, int] = (127, 127, 127)) -> Image.Image:
    rgba = np.array(pil_rgba.convert("RGBA")).astype(np.float32)
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3:4] / 255.0
    bg_arr = np.full_like(rgb, bg, dtype=np.float32)
    out = rgb * alpha + bg_arr * (1.0 - alpha)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _determine_max_patches(image: Image.Image) -> int:
    w, h = image.size
    max_val = (w // 16) * (h // 16)
    if max_val > 784:
        return 1024
    if max_val > 576:
        return 784
    if max_val > 256:
        return 576
    if max_val > 128:
        return 256
    return 128


def _load_on_device(device: str):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    with torch.no_grad():
        dummy_tokens = tokenizer(
            ["probe"],
            padding="max_length",
            truncation=True,
            max_length=196,
            return_tensors="pt",
        )
        dummy_tokens = _move_inputs_to_device(dummy_tokens, device)
        feat = model.get_text_features(**dummy_tokens, walk_type="long")
        vector_size = int(feat.shape[-1])
    return model, tokenizer, image_processor, vector_size


def load_clip_model() -> None:
    global _model, _tokenizer, _image_processor, _fusion_text_vec, _vector_size, _runtime_device
    if _model is not None:
        return

    device = DEFAULT_DEVICE
    try:
        model, tokenizer, image_processor, vector_size = _load_on_device(device)
    except RuntimeError as exc:
        if device != "cuda":
            raise
        print(f"CUDA load failed ({exc}). Falling back to CPU.")
        device = "cpu"
        model, tokenizer, image_processor, vector_size = _load_on_device(device)

    _model = model
    _tokenizer = tokenizer
    _image_processor = image_processor
    _vector_size = vector_size
    _runtime_device = device

    _fusion_text_vec = embed_text(settings.fusion_text)

    print(f"Model: {MODEL_ID}")
    print(f"Vector size: {_vector_size}")
    print(f"Device: {device}")


def unload_clip_model() -> None:
    global _model, _tokenizer, _image_processor, _fusion_text_vec, _vector_size, _runtime_device
    _model = None
    _tokenizer = None
    _image_processor = None
    _fusion_text_vec = None
    _vector_size = None
    _runtime_device = None


def get_vector_size() -> int:
    load_clip_model()
    return int(_vector_size)


def get_runtime_device() -> str:
    load_clip_model()
    return str(_runtime_device)


@torch.no_grad()
def embed_text(text: str) -> np.ndarray:
    load_clip_model()
    device = _runtime_device
    tokens = _tokenizer(
        [text.lower().strip()],
        padding="max_length",
        truncation=True,
        max_length=196,
        return_tensors="pt",
    )
    tokens = _move_inputs_to_device(tokens, device)
    feat = _model.get_text_features(**tokens, walk_type="long")
    feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    return feat[0].detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def embed_image(pil_rgba: Image.Image) -> np.ndarray:
    load_clip_model()
    device = _runtime_device
    if _detect_method_2(pil_rgba):
        pil_rgb = _apply_method_2(pil_rgba)
    else:
        pil_rgb = pil_rgba.convert("RGB")
    image_input = _image_processor(
        images=pil_rgb,
        max_num_patches=_determine_max_patches(pil_rgb),
        return_tensors="pt",
    )
    image_input = _move_inputs_to_device(image_input, device)
    feat = _model.get_image_features(**image_input)
    feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    return feat[0].detach().cpu().numpy().astype(np.float32)


def early_fusion_embed(pil_rgba: Image.Image) -> np.ndarray:
    load_clip_model()
    img_vec = embed_image(pil_rgba)
    fused = settings.fusion_weight_image * img_vec + settings.fusion_weight_text * _fusion_text_vec
    return _l2_normalize(fused)
```

- [ ] **Step 4.2: Smoke-check import**

Run: `cd backend && python -c "from app.services import clip_service; print('import OK')"`
Expected: `import OK`. This does NOT download the model — only imports the module.

(Do NOT call `load_clip_model()` in this step; that would trigger the multi-GB download. It will be exercised at runtime/seed time.)

- [ ] **Step 4.3: Commit**

```bash
git add backend/app/services/clip_service.py
git commit -m "feat(clip): swap MetaCLIP 2 for FG-CLIP 2 base + early-fusion helper

- AutoModelForCausalLM(trust_remote_code=True) with AutoTokenizer/AutoImageProcessor
- Text path uses walk_type='long' and max_length=196
- Image path applies notebook 'method 2' alpha compositing on transparent bg
- early_fusion_embed fuses 0.9*img + 0.1*fusion_text_vec then L2-normalizes"
```

---

## Task 5: Rewrite database.py to use embedded Qdrant

**Files:**
- Modify: `backend/app/database.py`

- [ ] **Step 5.1: Replace file contents**

```python
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
```

- [ ] **Step 5.2: Smoke-check import**

Run: `cd backend && python -c "from app import database; print('import OK')"`
Expected: `import OK` (no model download triggered — `get_vector_size` is only called inside `_ensure_qdrant_collection`, which runs at `connect_qdrant` time).

- [ ] **Step 5.3: Commit**

```bash
git add backend/app/database.py
git commit -m "feat(db): replace Milvus with embedded Qdrant (local file storage)

- connect_qdrant() opens QdrantClient(path=...); re-raises with a clear
  'storage locked' hint if the backend and seed conflict.
- _ensure_qdrant_collection creates a cosine collection sized to FG-CLIP 2
  and payload indexes on store/category for fast filter."
```

---

## Task 6: Create qdrant_service.py and delete milvus_service.py

**Files:**
- Create: `backend/app/services/qdrant_service.py`
- Delete: `backend/app/services/milvus_service.py`

- [ ] **Step 6.1: Create `backend/app/services/qdrant_service.py`**

```python
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
```

- [ ] **Step 6.2: Delete `backend/app/services/milvus_service.py`**

Run: `git rm backend/app/services/milvus_service.py`
Expected: file removed from tree and staged.

- [ ] **Step 6.3: Smoke-check import**

Run: `cd backend && python -c "from app.services.qdrant_service import upsert_product_embedding, search_similar_products; print('OK')"`
Expected: `OK`. Callers (`product_service.py`, `chat.py`, `seed_data.py`) still import from the old path — that's fixed in Task 7/8/10.

- [ ] **Step 6.4: Commit**

```bash
git add backend/app/services/qdrant_service.py
git commit -m "feat(qdrant_service): new vector service replacing milvus_service

- upsert_product_embedding now takes a precomputed embedding directly
  (seed computes from image instead of text)
- search_similar_products still embeds the query text internally, so
  product_service/chat keep the same call shape
- Uses UUIDv5 derived from product_id for stable idempotent upserts
- Only filters by store at the vector DB layer; category stays at Mongo"
```

---

## Task 7: Drop layer/description from product model + product service

**Files:**
- Modify: `backend/app/models/product.py`
- Modify: `backend/app/services/product_service.py`

- [ ] **Step 7.1: Replace `backend/app/models/product.py`**

```python
from pydantic import BaseModel
from bson import ObjectId


class PyObjectId(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return str(v)


class ProductCreate(BaseModel):
    name: str
    image_url: str
    store: str
    category: str


class ProductResponse(BaseModel):
    id: str
    name: str
    image_url: str
    store: str
    category: str

    class Config:
        populate_by_name = True


class ProductListResponse(BaseModel):
    items: list[ProductResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class FilterOptions(BaseModel):
    stores: list[str]
    categories: list[str]
```

- [ ] **Step 7.2: Replace `backend/app/services/product_service.py`**

```python
from typing import Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.models.product import FilterOptions, ProductListResponse, ProductResponse
from app.services.qdrant_service import search_similar_products

DEFAULT_STORES = ["Sample User"]


def _serialize(doc: dict) -> ProductResponse:
    doc["id"] = str(doc.pop("_id"))
    return ProductResponse(**doc)


async def _fetch_products_by_ids(
    db: AsyncIOMotorDatabase,
    ids: list[str],
    stores: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
) -> list[dict]:
    valid_object_ids = [ObjectId(product_id) for product_id in ids if ObjectId.is_valid(product_id)]
    if not valid_object_ids:
        return []

    mongo_query: dict = {"_id": {"$in": valid_object_ids}}
    if stores:
        mongo_query["store"] = {"$in": stores}
    if categories:
        mongo_query["category"] = {"$in": categories}

    docs = await db.products.find(mongo_query).to_list(length=len(valid_object_ids))
    doc_map = {str(doc["_id"]): doc for doc in docs}
    return [doc_map[product_id] for product_id in ids if product_id in doc_map]


async def get_products(
    db: AsyncIOMotorDatabase,
    page: int = 1,
    page_size: int = 12,
    search: Optional[str] = None,
    stores: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
    sort_by: str = "created",
    sort_order: str = "desc",
    semantic_search: bool = False,
) -> ProductListResponse:
    del semantic_search  # kept for API compat; actual semantic path triggers on `search`
    query: dict = {}

    if search:
        ids = search_similar_products(
            search,
            top_k=100,
            stores=stores,
        )
        skip = (page - 1) * page_size
        ordered_docs = await _fetch_products_by_ids(
            db,
            ids[skip : skip + page_size],
            stores=stores,
            categories=categories,
        )
        total = len(ids)
        items = [_serialize(doc) for doc in ordered_docs]
        return ProductListResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=max(1, (total + page_size - 1) // page_size),
        )

    if stores:
        query["store"] = {"$in": stores}
    if categories:
        query["category"] = {"$in": categories}

    sort_field_map = {
        "created": "_id",
        "name": "name",
    }
    mongo_sort_field = sort_field_map.get(sort_by, "_id")
    mongo_sort_dir = -1 if sort_order == "desc" else 1

    total = await db.products.count_documents(query)
    skip = (page - 1) * page_size

    cursor = db.products.find(query).sort(mongo_sort_field, mongo_sort_dir).skip(skip).limit(page_size)
    docs = await cursor.to_list(length=page_size)
    items = [_serialize(doc) for doc in docs]

    return ProductListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=max(1, (total + page_size - 1) // page_size),
    )


async def get_product_by_id(db: AsyncIOMotorDatabase, product_id: str) -> Optional[ProductResponse]:
    if not ObjectId.is_valid(product_id):
        return None
    doc = await db.products.find_one({"_id": ObjectId(product_id)})
    if not doc:
        return None
    return _serialize(doc)


async def get_filter_options(db: AsyncIOMotorDatabase) -> FilterOptions:
    stores = await db.products.distinct("store")
    categories = await db.products.distinct("category")

    return FilterOptions(
        stores=sorted(set(DEFAULT_STORES) | set(stores)),
        categories=sorted(categories),
    )


async def get_related_products(
    db: AsyncIOMotorDatabase, product_id: str, limit: int = 4
) -> list[ProductResponse]:
    product = await get_product_by_id(db, product_id)
    if not product:
        return []
    ids = search_similar_products(
        f"{product.name} {product.store} {product.category}",
        top_k=limit + 1,
    )
    ids = [i for i in ids if i != product_id][:limit]
    if not ids:
        cursor = (
            db.products.find({"category": product.category, "_id": {"$ne": ObjectId(product_id)}})
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        return [_serialize(doc) for doc in docs]
    docs = await _fetch_products_by_ids(db, ids)
    return [_serialize(doc) for doc in docs]
```

- [ ] **Step 7.3: Smoke-check imports**

Run: `cd backend && python -c "from app.services import product_service; from app.models.product import ProductResponse; print('OK')"`
Expected: `OK`.

- [ ] **Step 7.4: Commit**

```bash
git add backend/app/models/product.py backend/app/services/product_service.py
git commit -m "refactor: drop layer/description from product schema and service

- ProductResponse/ProductCreate/FilterOptions no longer carry layer or description
- product_service drops the layers filter parameter
- Store default is now 'Sample User' to match the seed CSV mapping
- Imports switched from milvus_service to qdrant_service"
```

---

## Task 8: Update products router and chat router

**Files:**
- Modify: `backend/app/routers/products.py`
- Modify: `backend/app/routers/chat.py`

- [ ] **Step 8.1: Replace `backend/app/routers/products.py`**

```python
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database import get_db
from app.models.product import FilterOptions, ProductListResponse, ProductResponse
from app.services import product_service

router = APIRouter(prefix="/api/products", tags=["products"])


@router.get("/filters", response_model=FilterOptions)
async def get_filter_options(db: AsyncIOMotorDatabase = Depends(get_db)):
    return await product_service.get_filter_options(db)


@router.get("", response_model=ProductListResponse)
async def list_products(
    page: int = Query(1, ge=1),
    page_size: int = Query(12, ge=1, le=100),
    search: Optional[str] = Query(None),
    stores: Optional[list[str]] = Query(None),
    categories: Optional[list[str]] = Query(None),
    sort_by: str = Query("created", pattern="^(created|name)$"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    semantic: bool = Query(False),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    return await product_service.get_products(
        db=db,
        page=page,
        page_size=page_size,
        search=search,
        stores=stores,
        categories=categories,
        sort_by=sort_by,
        sort_order=sort_order,
        semantic_search=semantic,
    )


@router.get("/{product_id}/related", response_model=list[ProductResponse])
async def get_related(
    product_id: str,
    limit: int = Query(4, ge=1, le=20),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    return await product_service.get_related_products(db, product_id, limit)


@router.get("/{product_id}", response_model=ProductResponse)
async def get_product(product_id: str, db: AsyncIOMotorDatabase = Depends(get_db)):
    product = await product_service.get_product_by_id(db, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product
```

- [ ] **Step 8.2: Update the import in `backend/app/routers/chat.py`**

Replace line 5 (`from app.services.milvus_service import search_similar_products`) with:

```python
from app.services.qdrant_service import search_similar_products
```

No other changes in `chat.py` — `search_similar_products(message, top_k=3)` still works.

- [ ] **Step 8.3: Smoke-check imports**

Run: `cd backend && python -c "from app.routers import products, chat; print('OK')"`
Expected: `OK`.

- [ ] **Step 8.4: Commit**

```bash
git add backend/app/routers/products.py backend/app/routers/chat.py
git commit -m "refactor(routers): drop layers query param; swap to qdrant_service

- /api/products no longer accepts layers[]=
- chat.py imports search_similar_products from qdrant_service"
```

---

## Task 9: Update main.py lifespan to use Qdrant

**Files:**
- Modify: `backend/app/main.py`

- [ ] **Step 9.1: Replace file contents**

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.database import connect_mongodb, connect_qdrant, disconnect_mongodb, disconnect_qdrant
from app.routers import chat, products
from app.services.clip_service import load_clip_model, unload_clip_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_mongodb()
    load_clip_model()
    connect_qdrant()
    yield
    await disconnect_mongodb()
    disconnect_qdrant()
    unload_clip_model()


app = FastAPI(
    title="OE-VLM Shop API",
    description="E-commerce API powered by FastAPI, MongoDB, and Qdrant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(products.router)
app.include_router(chat.router)

app.mount("/images", StaticFiles(directory="images"), name="images")


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 9.2: Smoke-check app import (no lifespan triggered)**

Run: `cd backend && python -c "from app.main import app; print('OK', len(app.routes))"`
Expected: prints `OK` and some number of routes (> 5).

- [ ] **Step 9.3: Commit**

```bash
git add backend/app/main.py
git commit -m "refactor(main): swap Milvus lifespan for Qdrant

connect_qdrant/disconnect_qdrant replace the Milvus connection hooks.
Load order unchanged: Mongo -> model (populates fusion_text_vec) -> Qdrant."
```

---

## Task 10: Rewrite seed_data.py for CSV + early-fusion seeding

**Files:**
- Modify: `backend/seed_data.py`

- [ ] **Step 10.1: Replace file contents**

```python
"""
Seed script: reads a CSV of products, embeds each image with FG-CLIP 2 early
fusion, and upserts to MongoDB + embedded Qdrant.

CSV columns (TIB format from the reference notebook):
    clipart_id, url, clipart_name, clipart_category_name
Extra columns are ignored. Rows missing `url` or `clipart_id` are skipped.

Mongo doc fields written:
    name         <- clipart_name
    image_url    <- url
    store        = "Sample User" (constant)
    category     <- clipart_category_name

Run (from the backend/ directory):

    python seed_data.py --csv ./data/products.csv
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm.auto import tqdm
from motor.motor_asyncio import AsyncIOMotorClient

from app.config import settings
from app.database import connect_qdrant, disconnect_qdrant, get_qdrant_client
from app.services.clip_service import early_fusion_embed, load_clip_model, unload_clip_model
from app.services.qdrant_service import upsert_product_embedding

STORE_CONSTANT = "Sample User"
DEFAULT_CSV = Path(__file__).parent / "data" / "products.csv"
DEFAULT_FAILURES = Path(__file__).parent / "failures.json"


def fetch_image(url: str, timeout: int = 30) -> Image.Image:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGBA")


def normalize_rows(df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for raw in df.to_dict(orient="records"):
        clipart_id = str(raw.get("clipart_id", "")).strip()
        url = str(raw.get("url", "")).strip()
        if not clipart_id or not url or url.lower() == "nan":
            continue
        rows.append({
            "clipart_id": clipart_id,
            "url": url,
            "name": str(raw.get("clipart_name", "")).strip() or "Untitled",
            "category": str(raw.get("clipart_category_name", "")).strip() or "Uncategorized",
        })
    return rows


async def seed(csv_path: Path, failures_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    rows = normalize_rows(df)
    print(f"Loaded {len(rows)} valid rows from {csv_path}")

    load_clip_model()
    connect_qdrant()
    client = AsyncIOMotorClient(settings.mongodb_url)
    db = client[settings.mongodb_db]

    await db.products.delete_many({})
    qdrant = get_qdrant_client()
    qdrant.delete_collection(collection_name=settings.qdrant_collection)
    # Recreate by reconnecting (ensures fresh collection with right schema)
    disconnect_qdrant()
    connect_qdrant()

    errors: list[dict] = []
    success = 0

    try:
        for row in tqdm(rows, desc="Seed (fetch + embed + upsert)"):
            try:
                img_rgba = fetch_image(row["url"])
                fused = early_fusion_embed(img_rgba)

                doc = {
                    "name": row["name"],
                    "image_url": row["url"],
                    "store": STORE_CONSTANT,
                    "category": row["category"],
                }
                result = await db.products.insert_one(doc)
                product_id = str(result.inserted_id)

                upsert_product_embedding(
                    product_id=product_id,
                    embedding=fused,
                    store=STORE_CONSTANT,
                    category=row["category"],
                )
                success += 1
            except Exception as e:
                errors.append({
                    "clipart_id": row.get("clipart_id", ""),
                    "url": row.get("url", ""),
                    "error": f"{type(e).__name__}: {e}",
                })
    finally:
        if errors:
            failures_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote {len(errors)} failures to {failures_path}")
        client.close()
        disconnect_qdrant()
        unload_clip_model()

    print(f"Seeded {success} products, {len(errors)} failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed products into MongoDB + Qdrant from CSV")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to the CSV file (TIB format)")
    parser.add_argument("--failures-out", type=Path, default=DEFAULT_FAILURES, help="Where to write failures.json")
    args = parser.parse_args()
    asyncio.run(seed(args.csv, args.failures_out))


if __name__ == "__main__":
    main()
```

- [ ] **Step 10.2: Smoke-check script imports (do NOT run the seed)**

Run: `cd backend && python -c "import seed_data; print('OK')"`
Expected: `OK`.

- [ ] **Step 10.3: Commit**

```bash
git add backend/seed_data.py
git commit -m "feat(seed): rewrite seed for CSV + FG-CLIP 2 early-fusion + Qdrant

- --csv CLI arg with default backend/data/products.csv
- TIB columns (clipart_id, url, clipart_name, clipart_category_name) mapped
  to product schema (store hardcoded to 'Sample User')
- Skip rows with missing url/clipart_id or failed fetch/embed; dump
  failures.json at end
- Wipes collections before reseeding to guarantee a clean run"
```

---

## Task 11: Drop layer + description from frontend types and API

**Files:**
- Modify: `frontend/src/types/index.ts`
- Modify: `frontend/src/lib/api.ts`

- [ ] **Step 11.1: Replace `frontend/src/types/index.ts`**

```ts
export interface Product {
  id: string;
  name: string;
  image_url: string;
  store: string;
  category: string;
}

export interface ProductListResponse {
  items: Product[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface FilterOptions {
  stores: string[];
  categories: string[];
}
```

- [ ] **Step 11.2: Edit `frontend/src/lib/api.ts`** — remove `layers?: string[]` from `ProductsParams`

Replace lines 5–15 with:

```ts
export interface ProductsParams {
  page?: number;
  page_size?: number;
  search?: string;
  stores?: string[];
  categories?: string[];
  sort_by?: string;
  sort_order?: string;
  semantic?: boolean;
}
```

- [ ] **Step 11.3: Verify TypeScript is happy after type change**

Run: `cd frontend && npx tsc --noEmit`
Expected: errors pointing at `ProductListPage.tsx`, `ProductCard.tsx`, `ProductDetailPage.tsx` (missing `layer`, `description`, `layers`). That's expected — Tasks 12–14 fix them.

Do not commit yet — these types are load-bearing and uncommitted frontend would leave the tree red. Bundle this with Tasks 12–14 in a single commit at the end of Task 14.

---

## Task 12: Drop layer badge from ProductCard

**Files:**
- Modify: `frontend/src/components/ProductCard.tsx`

- [ ] **Step 12.1: Replace the bottom info block (lines ~69–76)**

Replace:

```tsx
          <div className="flex items-center justify-between gap-2">
            <Badge variant="outline" className="max-w-[48%] truncate border-[#0d1b67]/15 bg-[#f4f7ff] text-[#0d1b67]">
              {product.store}
            </Badge>
            <Badge variant="outline" className="max-w-[48%] truncate border-[#2f6f55]/15 bg-[#edf8f2] text-[#2f6f55]">
              {product.layer}
            </Badge>
          </div>
```

with:

```tsx
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="max-w-full truncate border-[#0d1b67]/15 bg-[#f4f7ff] text-[#0d1b67]">
              {product.store}
            </Badge>
          </div>
```

---

## Task 13: Drop description section from ProductDetailPage

**Files:**
- Modify: `frontend/src/pages/ProductDetailPage.tsx`

- [ ] **Step 13.1: Remove the description block (lines ~104–109)**

Delete these lines:

```tsx
            <Separator />

            <div>
              <h2 className="text-sm font-semibold text-gray-900 mb-2">Mô tả</h2>
              <p className="text-sm text-gray-600 leading-relaxed">{product.description}</p>
            </div>
```

Leave the `<Separator />` above the "Quay lại danh sách" button removed along with the block — the button will sit directly under the category chip.

- [ ] **Step 13.2: Remove the now-unused `Separator` import if no other usage remains**

Run: `grep -n 'Separator' frontend/src/pages/ProductDetailPage.tsx`
Expected: only the `import { Separator }` line remains. If so, delete that import line:

```tsx
import { Separator } from "@/components/ui/separator";
```

If other `<Separator />` usages remain, leave the import.

---

## Task 14: Drop Layer filter UI, state, and toggles from ProductListPage

**Files:**
- Modify: `frontend/src/pages/ProductListPage.tsx`

- [ ] **Step 14.1a: Update `DEFAULT_STORE` constant (line 19)**

Change:

```tsx
const DEFAULT_STORE = "AnStore";
```

to:

```tsx
const DEFAULT_STORE = "Sample User";
```

This keeps the default-selected store matching the backend seed (Task 10 writes every product with `store = "Sample User"`). Otherwise the product list would render empty on first load because no product matches the default filter.

- [ ] **Step 14.1: Remove `selectedLayers` state (line ~37)**

Delete:

```tsx
  const selectedLayers = searchParams.getAll("layers");
```

- [ ] **Step 14.2: Remove `layers` from the products request (line ~77)**

Inside the `api.getProducts` call, delete this line:

```tsx
        layers: selectedLayers.length ? selectedLayers : undefined,
```

- [ ] **Step 14.3: Remove `toggleLayer` (lines ~109–114)**

Delete:

```tsx
  function toggleLayer(layer: string) {
    const next = selectedLayers.includes(layer)
      ? selectedLayers.filter((value) => value !== layer)
      : [...selectedLayers, layer];
    updateParams({ layers: next });
  }
```

- [ ] **Step 14.4: Remove layers from `hasActiveFilters` (line ~122)**

Change:

```tsx
  const hasActiveFilters =
    selectedCategories.length > 0 ||
    selectedLayers.length > 0 ||
    selectedStore !== DEFAULT_STORE;
```

to:

```tsx
  const hasActiveFilters =
    selectedCategories.length > 0 ||
    selectedStore !== DEFAULT_STORE;
```

- [ ] **Step 14.5: Remove the Layer filter section (lines ~154–170)**

Delete:

```tsx
      <Separator />

      {/* Layers */}
      {filterOptions && (
        <div>
          <h3 className="text-xs font-semibold text-[#0c1638] uppercase tracking-[0.06em] mb-3">Layer</h3>
          <div className="space-y-2">
            {filterOptions.layers.map((layer) => (
              <label key={layer} className="flex items-center gap-2 cursor-pointer group">
                <Checkbox
                  checked={selectedLayers.includes(layer)}
                  onCheckedChange={() => toggleLayer(layer)}
                />
                <span className="text-sm text-[#444956] group-hover:text-[#0d1b67]">{layer}</span>
              </label>
            ))}
          </div>
        </div>
      )}
```

The existing `<Separator />` between Stores and Categories stays — just remove the one introduced with the Layers block plus the block itself.

- [ ] **Step 14.6: Fix the mobile filter badge count (line ~247)**

Change the JSX:

```tsx
                <span className="bg-[#0d1b67] text-white text-xs rounded-full w-4 h-4 flex items-center justify-center">
                  (selectedStore !== DEFAULT_STORE ? 1 : 0) + selectedCategories.length + selectedLayers.length
                </span>
```

to:

```tsx
                <span className="bg-[#0d1b67] text-white text-xs rounded-full w-4 h-4 flex items-center justify-center">
                  {(selectedStore !== DEFAULT_STORE ? 1 : 0) + selectedCategories.length}
                </span>
```

(Note: the original had a JSX bug — the expression was rendered as literal text. This migration fixes it incidentally by wrapping in `{}` and removing the layers term.)

- [ ] **Step 14.7: Remove layer chips from the active-filters bar (lines ~278–283)**

Delete:

```tsx
            {selectedLayers.map((layer) => (
              <span key={layer} className="inline-flex items-center gap-1 bg-emerald-50 text-emerald-700 text-xs px-3 py-1 rounded-full border border-emerald-200">
                {layer}
                <button onClick={() => toggleLayer(layer)}><X className="h-3 w-3" /></button>
              </span>
            ))}
```

- [ ] **Step 14.8: Type-check the whole frontend**

Run: `cd frontend && npx tsc --noEmit`
Expected: clean pass (no errors).

- [ ] **Step 14.9: Dev-build smoke**

Run: `cd frontend && npm run build`
Expected: Vite builds without type errors. (Visual verification comes in Task 16.)

- [ ] **Step 14.10: Commit all frontend changes together**

```bash
git add frontend/src/types/index.ts frontend/src/lib/api.ts \
        frontend/src/components/ProductCard.tsx \
        frontend/src/pages/ProductDetailPage.tsx \
        frontend/src/pages/ProductListPage.tsx
git commit -m "feat(frontend): drop layer + description from UI

- Product/FilterOptions types no longer carry layer/description/layers
- api.ts ProductsParams no longer accepts layers[]
- ProductCard shows store badge only
- ProductDetailPage no longer renders description section
- ProductListPage drops Layer filter section, selectedLayers state,
  toggleLayer, and layer chips; fixes unrelated JSX bug in mobile
  filter-count badge while nearby"
```

---

## Task 15: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 15.1: Update the Stack bullet**

Replace:

```
- **Vector Search**: Milvus v2.4.13 with 384-dim cosine embeddings (`all-MiniLM-L6-v2`)
```

with:

```
- **Vector Search**: Qdrant embedded (local file storage) with 768-dim cosine embeddings from `qihoo360/fg-clip2-base` (FG-CLIP 2 base via `AutoModelForCausalLM(trust_remote_code=True)`)
```

- [ ] **Step 15.2: Update the "Infrastructure (Docker)" section**

Replace:

```bash
docker compose up -d    # Start MongoDB (27017), Milvus (19530), etcd, MinIO
docker compose down     # Stop all services
```

with:

```bash
docker compose up -d    # Start MongoDB (27017). Qdrant runs embedded in the backend process.
docker compose down     # Stop MongoDB
```

- [ ] **Step 15.3: Update the "Key Design Points" notes**

Replace the `Dual search modes`, `Startup lifecycle`, and `Embedding pipeline` bullets with:

```
**Search**: Products are retrieved via MongoDB filters or semantic text search through Qdrant. The `search` query string switches modes (when `search` is present, Qdrant returns a ranked ID list which is then resolved in Mongo).

**Startup lifecycle** (`backend/app/main.py`): The FastAPI lifespan context connects to MongoDB, loads the FG-CLIP 2 model (which also caches the fusion-text embedding), then opens the embedded Qdrant client. On shutdown they're released in reverse order.

**Embedding pipeline** (`backend/app/services/clip_service.py`): FG-CLIP 2 is lazy-loaded on first use. At seed time, the product image is fetched, preprocessed with "method 2" alpha compositing when it has a transparent background, embedded via `get_image_features`, and early-fused with the fixed prompt `"transparent background, isolated object"` (weights 0.9/0.1) before L2 normalization. At query time, only `embed_text` is called.

**Concurrency note**: The embedded Qdrant holds a file lock on `backend/qdrant_storage/`. Stop the backend (`uvicorn`) before running `seed_data.py`, and vice versa.
```

- [ ] **Step 15.4: Update the filter query params line**

Replace:

```
The `/api/products` endpoint supports: `page`, `page_size`, `search`, `brands[]`, `categories[]`, `min_price`, `max_price`, `sort_by`, `sort_order`, `semantic`.
```

with:

```
The `/api/products` endpoint supports: `page`, `page_size`, `search`, `stores[]`, `categories[]`, `sort_by`, `sort_order`, `semantic`.
```

- [ ] **Step 15.5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(CLAUDE.md): describe FG-CLIP 2 + embedded Qdrant architecture

Reflects that docker compose now starts only MongoDB, and that seeding
and runtime cannot hold the Qdrant storage lock concurrently."
```

---

## Task 16: End-to-end manual smoke test

**Files:** none modified

This task verifies the whole migration works. No commit — the goal is to confirm the preceding commits are correct.

- [ ] **Step 16.1: Start infrastructure**

Run: `docker compose up -d`
Expected: only `oe_vlm_mongo` container starts.

- [ ] **Step 16.2: Prepare a small seed CSV**

Create `backend/data/products.csv` with 5–10 rows (the developer should use their own sample; the format is `clipart_id,url,clipart_name,clipart_category_name`). Pick URLs that resolve publicly so the fetch step succeeds.

- [ ] **Step 16.3: Run the seed (backend NOT running)**

Run: `cd backend && source venv/bin/activate && python seed_data.py --csv ./data/products.csv`
Expected:
- Model download (first run only, large)
- Progress bar
- Final line `Seeded N products, M failed` with `N > 0`.
- `backend/qdrant_storage/` directory created.
- `backend/failures.json` exists only if any rows failed.

- [ ] **Step 16.4: Confirm Mongo and Qdrant contents**

Run: `cd backend && python -c "
from pymongo import MongoClient
from qdrant_client import QdrantClient
from app.config import settings
print('mongo:', MongoClient(settings.mongodb_url)[settings.mongodb_db].products.count_documents({}))
c = QdrantClient(path=settings.qdrant_path)
print('qdrant:', c.count(collection_name=settings.qdrant_collection).count)
c.close()
"`
Expected: both counts match the successful-row count from the seed.

- [ ] **Step 16.5: Start the backend**

Run: `cd backend && uvicorn app.main:app --reload --port 8000`
Expected: logs show `Connected to MongoDB`, model load info, and `Connected to Qdrant at ./qdrant_storage`.

- [ ] **Step 16.6: Hit the API**

In another shell:

```bash
curl -s localhost:8000/health | jq
curl -s 'localhost:8000/api/products?page=1&page_size=5' | jq '.items[] | {id, name, store, category}'
curl -s 'localhost:8000/api/products/filters' | jq
curl -s 'localhost:8000/api/products?search=red' | jq '.total, (.items[0] // {})'
```

Expected:
- `/health` → `{"status":"ok"}`
- `/api/products` → items have no `layer`/`description` fields.
- `/api/products/filters` → object with `stores` and `categories`, no `layers`.
- `/api/products?search=...` → total > 0 (assuming any product vaguely matches).

- [ ] **Step 16.7: Frontend smoke**

```bash
cd frontend && npm run dev
```

Open http://localhost:5173, visit `/products`:
- Left sidebar: only Store + Category filter groups (no "Layer").
- Product cards show only the store badge.
- Click any product: detail page has category chip, no "Mô tả" section.
- Type in the search box and submit: results update (semantic search is exercised).

- [ ] **Step 16.8: Failure-mode smoke**

Add one row to the CSV with a definitely-broken URL (`https://example.invalid/not-real.png`). Stop the backend. Re-run:

```
cd backend && python seed_data.py --csv ./data/products.csv
```

Expected:
- Seed completes with `failed >= 1`.
- `backend/failures.json` lists that row with a non-empty `error` message.

- [ ] **Step 16.9: Lock-error smoke (optional but recommended)**

Start the backend (`uvicorn ...`). In another shell, run `python seed_data.py --csv ./data/products.csv`.
Expected: `RuntimeError` with the message `"Qdrant storage '...' is locked or inaccessible. Stop the backend before seeding (or vice versa)..."`.

---

## Post-implementation checklist

- [ ] `docker compose ps` shows only `oe_vlm_mongo`
- [ ] `backend/qdrant_storage/` exists and is gitignored
- [ ] No file imports `app.services.milvus_service`
- [ ] No source file references `metaclip_model_id`, `MILVUS_*`, `pymilvus`
- [ ] `grep -rn 'product.layer\|product.description\|FilterOptions.layers\|layers?:' frontend/src` returns nothing
- [ ] `npx tsc --noEmit` passes in `frontend/`
- [ ] `python -c "from app.main import app"` succeeds in `backend/`

Once all are green, the migration is done.
