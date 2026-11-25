"""Gateway entry point: the single public door into the cache cluster."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.config import DEFAULT_SECRET, settings
from app.routes import health_routes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [gateway] %(name)s: %(message)s",
)
logger = logging.getLogger("gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hook. Later phases start the health monitor here and
    close the shared HTTP client on shutdown (graceful shutdown)."""
    app.state.started_at = time.time()
    if settings.cluster_secret == DEFAULT_SECRET:
        logger.warning("CLUSTER_SECRET is the default value - fine for local dev only")
    logger.info(
        "Gateway starting on %s:%d with %d configured node(s): %s",
        settings.host,
        settings.port,
        len(settings.nodes),
        ", ".join(n.node_id for n in settings.nodes),
    )
    yield
    logger.info("Gateway shutting down")


app = FastAPI(
    title="Distributed Cache - Gateway",
    description="Public API for the distributed in-memory cache cluster.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(health_routes.router)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port)