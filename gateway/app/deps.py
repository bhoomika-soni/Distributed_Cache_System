"""FastAPI dependencies: hand routes the objects created in main.py's lifespan."""
from __future__ import annotations

from fastapi import Request

from app.services.cache_service import CacheService
from app.services.node_manager import NodeManager


def get_cache_service(request: Request) -> CacheService:
    return request.app.state.cache_service


def get_node_manager(request: Request) -> NodeManager:
    return request.app.state.node_manager
