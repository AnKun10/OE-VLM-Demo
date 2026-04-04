from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routers import chat
from app.services.vlm_service import load_vlm_model, unload_vlm_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_vlm_model()
    yield
    unload_vlm_model()


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

app.mount("/images", StaticFiles(directory="images"), name="images")

@app.get("/health")
async def health():
    return {"status": "ok"}
