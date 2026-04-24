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
