from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import get_settings
from app.routers import calculations, health, iast


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    print(f"Starting Carbon Filter API in {settings.env} mode")
    print(f"CORS origins: {settings.cors_origins_list}")
    yield
    print("Shutting down Carbon Filter API")


app = FastAPI(
    title="Carbon Filter Sizing API",
    description="API for activated carbon filter dimensioning calculations",
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(calculations.router)
app.include_router(iast.router)


@app.get("/")
async def root():
    return {
        "name": "Carbon Filter Sizing API",
        "version": "1.0.0",
        "status": "running",
    }
