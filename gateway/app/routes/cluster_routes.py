"""Cluster information. Statuses are real: "unknown" until Phase 7 adds health checks."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.deps import get_node_manager
from app.schemas.cluster_schemas import ClusterResponse, NodeResponse
from app.services.node_manager import NodeManager

router = APIRouter(prefix="/cluster", tags=["cluster"])
Nodes = Annotated[NodeManager, Depends(get_node_manager)]


@router.get("", response_model=ClusterResponse)
async def cluster(nodes: Nodes):
    return ClusterResponse(node_count=len(nodes), node_ids=[n.node_id for n in nodes.all()])


@router.get("/nodes", response_model=list[NodeResponse])
async def list_nodes(nodes: Nodes):
    return [NodeResponse(node_id=n.node_id, url=n.url, status=n.status) for n in nodes.all()]


@router.get("/nodes/{node_id}", response_model=NodeResponse)
async def get_node(node_id: str, nodes: Nodes):
    node = nodes.get(node_id)
    if node is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Node '{node_id}' not found")
    return NodeResponse(node_id=node.node_id, url=node.url, status=node.status)
