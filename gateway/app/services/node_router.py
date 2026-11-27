"""Decides which node is responsible for a key, using a consistent hash ring.

The ring is built once from the node registry. Phase 6 uses `ring.get_nodes()`
to pick replicas, and Phase 7 passes it the set of healthy nodes so requests
skip failed nodes.
"""
from __future__ import annotations

from app.services.consistent_hash import HashRing
from app.services.node_manager import NodeInfo, NodeManager


class NodeRouter:
    def __init__(self, nodes: NodeManager, virtual_nodes: int) -> None:
        self._nodes = nodes
        self.ring = HashRing(virtual_nodes)
        for node in nodes.all():
            self.ring.add_node(node.node_id)

    def route(self, key: str) -> NodeInfo:
        """The primary node for `key`."""
        return self._nodes.get(self.ring.get_node(key))
