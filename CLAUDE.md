# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OE-VLM Shop** is a Vietnamese-language e-commerce demo for running shoes. It demonstrates vector-based semantic search using Milvus alongside standard MongoDB filtering.

## Development Commands

### Infrastructure (Docker)
```bash
docker compose up -d    # Start MongoDB (27017), Milvus (19530), etcd, MinIO
docker compose down     # Stop all services
```

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000   # Start dev server
python seed_data.py                          # Seed MongoDB + Milvus with sample products
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
- **Vector Search**: Milvus v2.4.13 with 384-dim cosine embeddings (`all-MiniLM-L6-v2`)
- **Frontend**: React 18 + Vite + TypeScript, TailwindCSS + ShadCN UI

### Key Design Points

**Dual search modes**: Products can be retrieved via standard MongoDB text/filter queries or semantic vector search through Milvus. The `semantic=true` query parameter switches modes. Milvus is optional — if unavailable, related-products falls back to category-based filtering.

**Startup lifecycle** (`backend/app/main.py`): The FastAPI lifespan context manager connects to both MongoDB and Milvus on startup, creating indexes if they don't exist.

**Embedding pipeline** (`backend/app/services/milvus_service.py`): SentenceTransformer is lazy-loaded on first use. Embeddings are generated from product text and stored in Milvus at seed time. Search uses `nprobe=16`, `IVF_FLAT` index.

**API proxy**: Vite dev server proxies `/api/*` to `http://localhost:8000`, so the frontend always uses relative `/api` paths.

**Environment config**: Backend reads from `backend/.env` (see `backend/.env.example`). Key variables: `MONGODB_URL`, `MILVUS_HOST`, `MILVUS_PORT`.

### Request Flow
1. Frontend (`frontend/src/lib/api.ts`) builds query strings and fetches `/api/products`
2. FastAPI router (`backend/app/routers/products.py`) parses params and calls service layer
3. `product_service.py` queries MongoDB with filters/pagination or delegates to `milvus_service.py` for semantic search
4. Results serialized via `ProductResponse` Pydantic models and returned as `ProductListResponse`

### Routes
| Frontend | Backend |
|----------|---------|
| `/` | `GET /api/products` (featured + new arrivals) |
| `/products` | `GET /api/products` (filtered list) |
| `/products/:id` | `GET /api/products/{id}` + `GET /api/products/{id}/related` |

The `/api/products` endpoint supports: `page`, `page_size`, `search`, `brands[]`, `categories[]`, `min_price`, `max_price`, `sort_by`, `sort_order`, `semantic`.
