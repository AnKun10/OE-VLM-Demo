from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db: str = "oe_vlm_shop"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "products"
    embedding_model: str = "all-MiniLM-L6-v2"

    class Config:
        env_file = ".env"


settings = Settings()
