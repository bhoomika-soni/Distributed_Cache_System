"""Cache node entry point. Each node is an independent FastAPI process."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.config import DEFAULT_SECRET, settings
from app.routes import internal_routes

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s %(levelname)-8s [{settings.node_id}] %(name)s: %(message)s",
)
logger = logging.getLogger("cache_node")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hook. Later phases create the cache engine and start the
    background TTL cleanup task here, and cancel it on shutdown."""
    app.state.started_at = time.time()
    if settings.cluster_secret == DEFAULT_SECRET:
        logger.warning("CLUSTER_SECRET is the default value - fine for local dev only")
    logger.info(
        "Node %s starting on %s:%d (capacity=%d keys)",
        settings.node_id,
        settings.host,
        settings.port,
        settings.capacity,
    )
    yield
    logger.info("Node %s shutting down", settings.node_id)


app = FastAPI(
    title=f"Distributed Cache - Node {settings.node_id}",
    description="Internal API of a single cache node. Not meant for public exposure.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(internal_routes.router)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port)