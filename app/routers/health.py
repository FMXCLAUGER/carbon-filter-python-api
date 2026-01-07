from fastapi import APIRouter

from app.services.pygaps_service import (
    is_pygaps_available,
    SUPPORTED_MODELS,
    DEFAULT_ISOTHERM_PARAMS,
)

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Health check endpoint for Railway/container orchestration."""
    return {"status": "healthy"}


@router.get("/api/status")
async def api_status():
    """Extended status with feature availability."""
    return {
        "status": "healthy",
        "features": {
            "pygaps_available": is_pygaps_available(),
            "supported_isotherm_models": SUPPORTED_MODELS,
            "default_pollutants": list(DEFAULT_ISOTHERM_PARAMS.keys()),
        },
    }
