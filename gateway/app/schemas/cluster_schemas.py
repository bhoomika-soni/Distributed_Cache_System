"""Public API models for cluster information."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class NodeResponse(BaseModel):
    node_id: str
    url: str
    status: Literal["unknown", "healthy", "unhealthy", "recovering"]
    consecutive_failures: int
    last_checked_at: float | None
    last_success_at: float | None
    last_error: str | None
    last_status_change_at: float | None
    resync_attempts: int
    last_resync: dict | None  # outcome of the latest re-sync (copied / already_current / error)


class ClusterResponse(BaseModel):
    node_count: int
    node_ids: list[str]


class ClusterHealthResponse(BaseModel):
    # healthy: every node healthy | degraded: some are | unavailable: none are
    status: Literal["healthy", "degraded", "unavailable"]
    healthy_nodes: int
    total_nodes: int
    nodes: dict[str, str]  # node_id -> status
