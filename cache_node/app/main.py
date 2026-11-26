"""Cache node entry point. Each node is an independent FastAPI process."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.cache.cache_engine import CacheEngine
from app.cache.ttl_manager import TTLCleaner
from app.config import DEFAULT_SECRET, settings
from app.routes import internal_routes

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s %(levelname)-8s [{settings.node_id}] %(name)s: %(message)s",
)
logger = logging.getLogger("cache_node")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the cache engine and start the background TTL cleaner on startup;
    stop the cleaner on shutdown (graceful shutdown)."""
    app.state.started_at = time.time()
    app.state.engine = CacheEngine(capacity=settings.capacity)
    cleaner = TTLCleaner(app.state.engine, interval=settings.ttl_cleanup_interval)
    cleaner.start()
    app.state.cleaner = cleaner

    if settings.cluster_secret == DEFAULT_SECRET:
        logger.warning("CLUSTER_SECRET is the default value - fine for local dev only")
    logger.info(
        "Node %s starting on %s:%d (capacity=%d keys, cleanup every %ss)",
        settings.node_id,
        settings.host,
        settings.port,
        settings.capacity,
        settings.ttl_cleanup_interval,
    )
    try:
        yield
    finally:
        await cleaner.stop()
        logger.info("Node %s shut down", settings.node_id)


app = FastAPI(
    title=f"Distributed Cache - Node {settings.node_id}",
    description="Internal API of a single cache node. Not meant for public exposure.",
    version="0.2.0",
    lifespan=lifespan,
)
app.include_router(internal_routes.router)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Safety net: the engine rejects bad input with ValueError. Pydantic should
    catch it first (422), but if anything slips through, answer 400 not 500."""
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port)
