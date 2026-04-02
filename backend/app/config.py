from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db: str = "fashion_db"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "products"
    metaclip_model_id: str = "facebook/metaclip-2-mt5-worldwide-b32"

    class Config:
        env_file = ".env"


settings = Settings()
