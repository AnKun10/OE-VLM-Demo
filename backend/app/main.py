from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.database import connect_mongodb, connect_qdrant, disconnect_mongodb, disconnect_qdrant
from app.routers import chat, products
from app.services.clip_service import load_clip_model, unload_clip_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_mongodb()
    load_clip_model()
    connect_qdrant()
    yield
    await disconnect_mongodb()
    disconnect_qdrant()
    unload_clip_model()


app = FastAPI(
    title="OE-VLM Shop API",
    description="E-commerce API powered by FastAPI, MongoDB, and Qdrant",
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

app.include_router(products.router)
app.include_router(chat.router)

app.mount("/images", StaticFiles(directory="images"), name="images")


@app.get("/health")
async def health():
    return {"status": "ok"}
