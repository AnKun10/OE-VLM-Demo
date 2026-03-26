from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import connect_mongodb, disconnect_mongodb, connect_milvus, disconnect_milvus
from app.routers import products, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_mongodb()
    connect_milvus()
    yield
    await disconnect_mongodb()
    disconnect_milvus()


app = FastAPI(
    title="OE-VLM Shop API",
    description="E-commerce API powered by FastAPI, MongoDB, and Milvus",
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


@app.get("/health")
async def health():
    return {"status": "ok"}
