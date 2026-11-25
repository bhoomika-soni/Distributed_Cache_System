from __future__ import annotations

import time

from fastapi import APIRouter, Request

from app.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict:
    """Gateway liveness. Says nothing about node health (that comes in Phase 7)."""
    return {
        "status": "ok",
        "service": "gateway",
        "uptime_seconds": round(time.time() - request.app.state.started_at, 2),
        "configured_nodes": len(settings.nodes),
    }