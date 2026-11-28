"""Cluster information: registry, per-node health, overall cluster health."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.deps import get_node_manager
from app.schemas.cluster_schemas import ClusterHealthResponse, ClusterResponse, NodeResponse
from app.services.node_manager import HEALTHY, NodeInfo, NodeManager

router = APIRouter(prefix="/cluster", tags=["cluster"])
Nodes = Annotated[NodeManager, Depends(get_node_manager)]


def _node_response(n: NodeInfo) -> NodeResponse:
    return NodeResponse(
        node_id=n.node_id,
        url=n.url,
        status=n.status,
        consecutive_failures=n.consecutive_failures,
        last_checked_at=n.last_checked_at,
        last_success_at=n.last_success_at,
        last_error=n.last_error,
        last_status_change_at=n.last_status_change_at,
        resync_attempts=n.resync_attempts,
        last_resync=n.last_resync,
    )


@router.get("", response_model=ClusterResponse)
async def cluster(nodes: Nodes):
    return ClusterResponse(node_count=len(nodes), node_ids=[n.node_id for n in nodes.all()])


@router.get("/health", response_model=ClusterHealthResponse)
async def cluster_health(nodes: Nodes, response: Response):
    """200 when healthy or degraded; 503 when no node is healthy."""
    all_nodes = nodes.all()
    healthy = sum(1 for n in all_nodes if n.status == HEALTHY)
    if healthy == len(all_nodes):
        overall = "healthy"
    elif healthy == 0:
        overall = "unavailable"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        overall = "degraded"
    return ClusterHealthResponse(
        status=overall,
        healthy_nodes=healthy,
        total_nodes=len(all_nodes),
        nodes={n.node_id: n.status for n in all_nodes},
    )


@router.get("/nodes", response_model=list[NodeResponse])
async def list_nodes(nodes: Nodes):
    return [_node_response(n) for n in nodes.all()]


@router.get("/nodes/{node_id}", response_model=NodeResponse)
async def get_node(node_id: str, nodes: Nodes):
    node = nodes.get(node_id)
    if node is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Node '{node_id}' not found")
    return _node_response(node)
