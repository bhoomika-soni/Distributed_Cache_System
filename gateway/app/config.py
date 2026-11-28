"""Gateway configuration.

All settings come from environment variables with safe local-dev defaults.
Settings are validated once at startup so a bad value fails fast with a clear
message instead of causing odd behaviour later.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_SECRET = "change-me-in-production"
DEFAULT_NODES = (
    "node-1=http://localhost:8001,"
    "node-2=http://localhost:8002,"
    "node-3=http://localhost:8003"
)


@dataclass(frozen=True)
class NodeConfig:
    """A cache node the gateway knows about (static registry for now)."""

    node_id: str
    url: str


def parse_nodes(raw: str) -> list[NodeConfig]:
    """Parse 'node-1=http://host:port,node-2=...' into NodeConfig objects."""
    nodes: list[NodeConfig] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        node_id, sep, url = item.partition("=")
        if not sep or not node_id.strip() or not url.strip():
            raise ValueError(
                f"Invalid NODES entry {item!r}; expected 'node_id=http://host:port'"
            )
        nodes.append(NodeConfig(node_id.strip(), url.strip().rstrip("/")))

    ids = [n.node_id for n in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate node ids in NODES")
    return nodes


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    cluster_secret: str
    nodes: list[NodeConfig]
    replication_factor: int
    virtual_nodes: int
    health_check_interval: float  # seconds between health probes
    request_timeout: float  # seconds, per gateway -> node request
    max_retries: int  # extra attempts after the first one (kept small on purpose)
    failure_threshold: int  # consecutive failures before a node is marked unhealthy
    recovery_threshold: int  # consecutive successful probes before a node starts recovering


def load_settings() -> Settings:
    """Read and validate settings from the environment."""
    nodes = parse_nodes(os.getenv("NODES", DEFAULT_NODES))
    settings = Settings(
        host=os.getenv("GATEWAY_HOST", "0.0.0.0"),
        port=int(os.getenv("GATEWAY_PORT", "8000")),
        cluster_secret=os.getenv("CLUSTER_SECRET", DEFAULT_SECRET),
        nodes=nodes,
        replication_factor=int(os.getenv("REPLICATION_FACTOR", "3")),
        virtual_nodes=int(os.getenv("VIRTUAL_NODES", "100")),
        health_check_interval=float(os.getenv("HEALTH_CHECK_INTERVAL", "5")),
        request_timeout=float(os.getenv("REQUEST_TIMEOUT", "2.0")),
        max_retries=int(os.getenv("MAX_RETRIES", "2")),
        failure_threshold=int(os.getenv("FAILURE_THRESHOLD", "2")),
        recovery_threshold=int(os.getenv("RECOVERY_THRESHOLD", "2")),
    )

    if not 1 <= settings.port <= 65535:
        raise ValueError(f"GATEWAY_PORT out of range: {settings.port}")
    if not nodes:
        raise ValueError("NODES must list at least one cache node")
    if not 1 <= settings.replication_factor <= len(nodes):
        raise ValueError(
            f"REPLICATION_FACTOR must be between 1 and the number of nodes "
            f"({len(nodes)}), got {settings.replication_factor}"
        )
    if settings.virtual_nodes < 1:
        raise ValueError("VIRTUAL_NODES must be >= 1")
    if settings.health_check_interval <= 0 or settings.request_timeout <= 0:
        raise ValueError("HEALTH_CHECK_INTERVAL and REQUEST_TIMEOUT must be > 0")
    if settings.max_retries < 0:
        raise ValueError("MAX_RETRIES must be >= 0")
    if settings.failure_threshold < 1 or settings.recovery_threshold < 1:
        raise ValueError("FAILURE_THRESHOLD and RECOVERY_THRESHOLD must be >= 1")
    return settings


settings = load_settings()
