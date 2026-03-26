from typing import Optional
from pydantic import BaseModel, Field
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


class ProductColor(BaseModel):
    name: str
    hex: str
    image_url: Optional[str] = None


class ProductCreate(BaseModel):
    name: str
    brand: str
    category: str
    description: str
    price: float
    original_price: Optional[float] = None
    colors: list[ProductColor] = []
    sizes: list[str] = []
    images: list[str] = []
    tags: list[str] = []
    is_new: bool = False
    in_stock: bool = True
    stock_qty: int = 100
    rating: float = 0.0
    review_count: int = 0

    @property
    def discount_percent(self) -> Optional[int]:
        if self.original_price and self.original_price > self.price:
            return round((1 - self.price / self.original_price) * 100)
        return None


class ProductResponse(BaseModel):
    id: str
    name: str
    brand: str
    category: str
    description: str
    price: float
    original_price: Optional[float] = None
    discount_percent: Optional[int] = None
    colors: list[ProductColor] = []
    sizes: list[str] = []
    images: list[str] = []
    tags: list[str] = []
    is_new: bool = False
    in_stock: bool = True
    stock_qty: int = 100
    rating: float = 0.0
    review_count: int = 0

    class Config:
        populate_by_name = True


class ProductListResponse(BaseModel):
    items: list[ProductResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class FilterOptions(BaseModel):
    brands: list[str]
    categories: list[str]
    min_price: float
    max_price: float
