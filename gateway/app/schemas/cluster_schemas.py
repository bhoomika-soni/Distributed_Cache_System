"""Public API models for cluster information."""
from __future__ import annotations

from pydantic import BaseModel


class NodeResponse(BaseModel):
    node_id: str
    url: str
    status: str  # "unknown" until health monitoring exists (Phase 7)


class ClusterResponse(BaseModel):
    node_count: int
    node_ids: list[str]
