from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db: str = "fashion_db"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "products"
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"

    class Config:
        env_file = ".env"


settings = Settings()
