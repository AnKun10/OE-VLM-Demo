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
