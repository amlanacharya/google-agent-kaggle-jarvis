"""Health check endpoints."""

from fastapi import APIRouter
from datetime import datetime
from typing import Dict, Any

router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check - verifies all dependencies are ready."""
    checks = {
        "redis": "unknown",
        "mongodb": "unknown",
        "chromadb": "unknown",
    }

    # TODO: Implement actual dependency checks
    # For now, return basic status

    all_ready = all(status == "healthy" for status in checks.values())

    return {
        "status": "ready" if all_ready else "not_ready",
        "timestamp": datetime.utcnow().isoformat(),
        "dependencies": checks,
    }


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """Liveness check - verifies application is running."""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
    }
