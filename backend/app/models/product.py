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


class ProductCreate(BaseModel):
    name: str
    image_url: str
    store: str
    layer: str
    category: str
    description: str


class ProductResponse(BaseModel):
    id: str
    name: str
    image_url: str
    store: str
    layer: str
    category: str
    description: str

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
    layers: list[str]
