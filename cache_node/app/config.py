"""Cache node configuration, read from environment variables.

Every node runs the same code; only these variables differ per node
(NODE_ID and NODE_PORT in particular).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_SECRET = "change-me-in-production"


@dataclass(frozen=True)
class Settings:
    node_id: str
    host: str
    port: int
    capacity: int  # max number of keys before LRU eviction kicks in
    ttl_cleanup_interval: float  # seconds between background expiry sweeps
    cluster_secret: str


def load_settings() -> Settings:
    settings = Settings(
        node_id=os.getenv("NODE_ID", "node-1"),
        host=os.getenv("NODE_HOST", "0.0.0.0"),
        port=int(os.getenv("NODE_PORT", "8001")),
        capacity=int(os.getenv("CACHE_CAPACITY", "1000")),
        ttl_cleanup_interval=float(os.getenv("TTL_CLEANUP_INTERVAL", "1.0")),
        cluster_secret=os.getenv("CLUSTER_SECRET", DEFAULT_SECRET),
    )

    if not settings.node_id.strip():
        raise ValueError("NODE_ID must not be empty")
    if not 1 <= settings.port <= 65535:
        raise ValueError(f"NODE_PORT out of range: {settings.port}")
    if settings.capacity < 1:
        raise ValueError("CACHE_CAPACITY must be >= 1")
    if settings.ttl_cleanup_interval <= 0:
        raise ValueError("TTL_CLEANUP_INTERVAL must be > 0")
    return settings


settings = load_settings()