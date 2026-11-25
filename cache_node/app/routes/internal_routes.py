from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

from app.config import settings
from app.security import require_cluster_secret

# Every route on this router requires the cluster secret.
router = APIRouter(
    prefix="/internal",
    tags=["internal"],
    dependencies=[Depends(require_cluster_secret)],
)


@router.get("/health")
async def internal_health(request: Request) -> dict:
    """Node health, polled by the gateway's health monitor (Phase 7)."""
    return {
        "status": "up",
        "node_id": settings.node_id,
        "host": settings.host,
        "port": settings.port,
        "capacity": settings.capacity,
        "uptime_seconds": round(time.time() - request.app.state.started_at, 2),
    }