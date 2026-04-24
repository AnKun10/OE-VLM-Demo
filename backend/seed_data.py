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
