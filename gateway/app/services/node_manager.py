"""Registry of the cache nodes the gateway knows about."""
from __future__ import annotations

from dataclasses import dataclass

from app.config import NodeConfig


@dataclass
class NodeInfo:
    node_id: str
    url: str
    # "unknown" until the health monitor exists (Phase 7). We deliberately do
    # not claim "healthy" for a node we have never checked.
    status: str = "unknown"


class NodeManager:
    """Static registry built from the NODES setting.

    Phase 7 adds status tracking (healthy/unhealthy) on top of this.
    """

    def __init__(self, configs: list[NodeConfig]) -> None:
        self._nodes: dict[str, NodeInfo] = {
            c.node_id: NodeInfo(node_id=c.node_id, url=c.url) for c in configs
        }

    def all(self) -> list[NodeInfo]:
        return list(self._nodes.values())

    def get(self, node_id: str) -> NodeInfo | None:
        return self._nodes.get(node_id)

    def __len__(self) -> int:
        return len(self._nodes)
