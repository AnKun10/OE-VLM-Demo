from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.models.vlm import VLMManager
from app.routers import chat, files
from app.services.image_compressor import ImageCompressorEngine
from app.services.image_compressor.cache import CaptionCache


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager = VLMManager()
    manager.load()
    app.state.vlm_manager = manager

    cfg = manager.compressor_config()
    if cfg:
        cache = CaptionCache(cfg["cache_db_path"])
        await cache.init()
        app.state.compressor_engine = ImageCompressorEngine(
            cache=cache, vlm_manager=manager,
            caption_model_id=cfg["caption_model_id"],
            router_model_id=cfg["router_model_id"],
            webui_internal_base=cfg.get("webui_internal_base", "http://127.0.0.1:8000"),
            caption_max_tokens=cfg.get("caption_max_tokens", 80),
            router_max_tokens=cfg.get("router_max_tokens", 60),
            caption_timeout_s=cfg.get("caption_timeout_s", 30),
            router_timeout_s=cfg.get("router_timeout_s", 15),
            router_failopen_keep=cfg.get("router_failopen_keep", True),
        )
        print(f"[lifespan] compressor enabled "
              f"(caption={cfg['caption_model_id']}, router={cfg['router_model_id']})")
    else:
        app.state.compressor_engine = None
        print("[lifespan] compressor disabled (no `compressor:` yaml block)")
    yield


app = FastAPI(
    title="OE-VLM Shop API",
    description="E-commerce Chatbot API powered by FastAPI and VLM",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(files.router)

app.mount("/images", StaticFiles(directory="images"), name="images")


@app.get("/health")
async def health():
    return {"status": "ok"}
