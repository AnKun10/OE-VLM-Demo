from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    vlm_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vlm_device: str = "auto"

    class Config:
        env_file = ".env"


settings = Settings()
