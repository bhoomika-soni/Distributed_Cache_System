"""Decides which node is responsible for a key.

Phase 4: every key goes to the FIRST configured node. That is intentionally
simple - it gives us a working end-to-end path. Phase 5 replaces the body of
`route()` with a consistent-hash ring; nothing else in the gateway changes,
because everything calls through this one method.
"""
from __future__ import annotations

from app.services.node_manager import NodeInfo, NodeManager


class NodeRouter:
    def __init__(self, nodes: NodeManager) -> None:
        self._nodes = nodes

    def route(self, key: str) -> NodeInfo:
        return self._nodes.all()[0]
