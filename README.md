# OE-VLM Shop

E-commerce site for running shoes built with FastAPI, MongoDB, Milvus, React, TailwindCSS, and ShadCN UI.

## Stack

| Layer    | Tech                                |
|----------|-------------------------------------|
| Backend  | FastAPI (Python 3.11+)              |
| Metadata | MongoDB (via Motor async driver)    |
| Vectors  | Milvus (semantic search)            |
| Frontend | React 18 + Vite + TailwindCSS + ShadCN UI |

## Pages

- **Home** — Hero banner, brand filter links, featured & new products, CTA
- **Product List** — Grid with search bar, brand/category/price filters, sort, pagination
- **Product Detail** — Image gallery, color/size selector, quantity, related products

## Quick Start

### 1. Start databases

```bash
docker compose up -d
```

Wait ~30 seconds for Milvus to be ready.

### 2. Backend

```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

# Seed sample data
python seed_data.py

# Start API server
uvicorn app.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/products` | List products with filters |
| GET | `/api/products/{id}` | Get product details |
| GET | `/api/products/{id}/related` | Get related products via Milvus |
| GET | `/api/products/filters` | Get filter options (brands, categories, price range) |

### Query params for `GET /api/products`

| Param | Type | Description |
|-------|------|-------------|
| `page` | int | Page number (default: 1) |
| `page_size` | int | Items per page (default: 12) |
| `search` | string | Text search |
| `brands` | string[] | Filter by brand(s) |
| `categories` | string[] | Filter by category |
| `min_price` | float | Minimum price |
| `max_price` | float | Maximum price |
| `sort_by` | string | `created`, `price`, `name`, `rating`, `discount` |
| `sort_order` | string | `asc` or `desc` |
| `semantic` | bool | Use Milvus vector search (default: false) |

## Architecture

```
frontend/               # React + Vite app
  src/
    pages/              # HomePage, ProductListPage, ProductDetailPage
    components/         # Navbar, Footer, ProductCard, UI primitives
    lib/                # api.ts (fetch client), utils.ts
    types/              # TypeScript interfaces

backend/
  app/
    main.py             # FastAPI app + lifespan (MongoDB + Milvus connect)
    config.py           # Pydantic settings (from .env)
    database.py         # MongoDB Motor + Milvus connection helpers
    models/product.py   # Pydantic schemas
    routers/products.py # API routes
    services/
      product_service.py   # MongoDB CRUD + aggregation
      milvus_service.py    # Embeddings + vector search
  seed_data.py          # Sample shoe data seeder
```

## Notes

- Milvus is optional — the app falls back gracefully if Milvus is unavailable (related products will use MongoDB category fallback).
- Semantic search (`?semantic=true`) requires Milvus to be running and products to be seeded.
- The embedding model (`all-MiniLM-L6-v2`) is downloaded automatically on first run via `sentence-transformers`.
