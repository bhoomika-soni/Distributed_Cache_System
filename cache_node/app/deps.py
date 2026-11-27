"""FastAPI dependencies shared by the routes."""
from __future__ import annotations

from fastapi import Request

from app.cache.cache_engine import CacheEngine
from app.replication.replication_manager import ReplicationManager


def get_engine(request: Request) -> CacheEngine:
    """The node's single CacheEngine, created in main.py's lifespan hook."""
    return request.app.state.engine


def get_replication(request: Request) -> ReplicationManager:
    """The node's ReplicationManager, created in main.py's lifespan hook."""
    return request.app.state.replication
