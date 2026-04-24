# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OE-VLM Shop** is a Vietnamese-language e-commerce demo for running shoes. It demonstrates vector-based semantic search using Qdrant alongside standard MongoDB filtering.

## Development Commands

### Infrastructure (Docker)
```bash
docker compose up -d    # Start MongoDB (27017). Qdrant runs embedded in the backend process.
docker compose down     # Stop MongoDB
```

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000           # Start dev server
python seed_data.py --csv ./data/products.csv       # Seed MongoDB + Qdrant from a CSV
```

### Frontend
```bash
cd frontend
npm install
npm run dev      # Vite dev server at http://localhost:5173
npm run build    # tsc + vite build
```

No test or lint commands are configured.

## Architecture

### Stack
- **Backend**: FastAPI + Uvicorn (Python 3.11+)
- **Metadata DB**: MongoDB 7 via Motor (async)
- **Vector Search**: Qdrant embedded (local file storage) with 768-dim cosine embeddings from `qihoo360/fg-clip2-base` (FG-CLIP 2 base via `AutoModelForCausalLM(trust_remote_code=True)`)
- **Frontend**: React 18 + Vite + TypeScript, TailwindCSS + ShadCN UI

### Key Design Points

**Search**: Products are retrieved via MongoDB filters or semantic text search through Qdrant. The `search` query string switches modes (when `search` is present, Qdrant returns a ranked ID list which is then resolved in Mongo).

**Startup lifecycle** (`backend/app/main.py`): The FastAPI lifespan context connects to MongoDB, loads the FG-CLIP 2 model (which also caches the fusion-text embedding), then opens the embedded Qdrant client. On shutdown they're released in reverse order.

**Embedding pipeline** (`backend/app/services/clip_service.py`): FG-CLIP 2 is lazy-loaded on first use. At seed time, the product image is fetched, preprocessed with "method 2" alpha compositing when it has a transparent background, embedded via `get_image_features`, and early-fused with the fixed prompt `"transparent background, isolated object"` (weights 0.9/0.1) before L2 normalization. At query time, only `embed_text` is called.

**Concurrency note**: The embedded Qdrant holds a file lock on `backend/qdrant_storage/`. Stop the backend (`uvicorn`) before running `seed_data.py`, and vice versa.

**API proxy**: Vite dev server proxies `/api/*` to `http://localhost:8000`, so the frontend always uses relative `/api` paths.

**Environment config**: Backend reads from `backend/.env` (see `backend/.env.example`). Key variables: `MONGODB_URL`, `QDRANT_PATH`, `FGCLIP_MODEL_ID`.

### Request Flow
1. Frontend (`frontend/src/lib/api.ts`) builds query strings and fetches `/api/products`
2. FastAPI router (`backend/app/routers/products.py`) parses params and calls service layer
3. `product_service.py` queries MongoDB with filters/pagination or delegates to `qdrant_service.py` for semantic search
4. Results serialized via `ProductResponse` Pydantic models and returned as `ProductListResponse`

### Routes
| Frontend | Backend |
|----------|---------|
| `/` | `GET /api/products` (featured + new arrivals) |
| `/products` | `GET /api/products` (filtered list) |
| `/products/:id` | `GET /api/products/{id}` + `GET /api/products/{id}/related` |

The `/api/products` endpoint supports: `page`, `page_size`, `search`, `stores[]`, `categories[]`, `sort_by`, `sort_order`, `semantic`.
