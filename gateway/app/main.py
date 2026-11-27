"""Gateway entry point: the single public door into the cache cluster."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.config import DEFAULT_SECRET, settings
from app.routes import cache_routes, cluster_routes, health_routes
from app.services.cache_service import CacheService
from app.services.node_client import (
    NodeClient,
    NodeResponseError,
    NodeTimeoutError,
    NodeUnavailableError,
)
from app.services.node_manager import NodeManager
from app.services.node_router import NodeRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [gateway] %(name)s: %(message)s",
)
logger = logging.getLogger("gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the gateway's objects on startup and close the HTTP client on shutdown."""
    app.state.started_at = time.time()
    if settings.cluster_secret == DEFAULT_SECRET:
        logger.warning("CLUSTER_SECRET is the default value - fine for local dev only")

    # ONE shared client for the whole process: it keeps connections to the
    # nodes open (keep-alive) instead of reconnecting on every request.
    http = httpx.AsyncClient(
        timeout=settings.request_timeout,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    node_manager = NodeManager(settings.nodes)
    node_client = NodeClient(http, settings.cluster_secret, settings.max_retries)
    app.state.node_manager = node_manager
    app.state.cache_service = CacheService(NodeRouter(node_manager), node_client)

    logger.info(
        "Gateway starting on %s:%d with %d configured node(s): %s",
        settings.host, settings.port, len(node_manager),
        ", ".join(n.node_id for n in node_manager.all()),
    )
    try:
        yield
    finally:
        await http.aclose()
        logger.info("Gateway shut down")


app = FastAPI(
    title="Distributed Cache - Gateway",
    description="Public API for the distributed in-memory cache cluster.",
    version="0.4.0",
    lifespan=lifespan,
)
app.include_router(health_routes.router)
app.include_router(cache_routes.router)
app.include_router(cluster_routes.router)


# ---- Node failures -> HTTP status codes (the client sees a clear, honest error)
# 503: node unreachable   504: node too slow   502: node answered with garbage/401/5xx
@app.exception_handler(NodeTimeoutError)
async def node_timeout_handler(request: Request, exc: NodeTimeoutError) -> JSONResponse:
    logger.warning("%s", exc)
    return JSONResponse(status_code=status.HTTP_504_GATEWAY_TIMEOUT, content={"detail": str(exc)})


@app.exception_handler(NodeUnavailableError)
async def node_unavailable_handler(request: Request, exc: NodeUnavailableError) -> JSONResponse:
    logger.warning("%s", exc)
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": str(exc)})


@app.exception_handler(NodeResponseError)
async def node_response_handler(request: Request, exc: NodeResponseError) -> JSONResponse:
    # e.g. 401 means gateway and node disagree about CLUSTER_SECRET - a config bug.
    logger.error("%s", exc)
    return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content={"detail": str(exc)})


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port)
