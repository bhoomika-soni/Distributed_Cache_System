"""Decides which node is responsible for a key, using a consistent hash ring.

The ring is built once from the node registry. `replica_set()` uses
`ring.get_nodes()` to pick the primary plus its replicas; Phase 7 passes it the
set of healthy nodes so requests skip failed nodes.
"""
from __future__ import annotations

from app.services.consistent_hash import HashRing
from app.services.node_manager import NodeInfo, NodeManager


class NodeRouter:
    def __init__(self, nodes: NodeManager, virtual_nodes: int, replication_factor: int = 1) -> None:
        self._nodes = nodes
        self.replication_factor = replication_factor
        self.ring = HashRing(virtual_nodes)
        for node in nodes.all():
            self.ring.add_node(node.node_id)

    def route(self, key: str) -> NodeInfo:
        """The primary node for `key` (where reads go)."""
        return self._nodes.get(self.ring.get_node(key))

    def replica_set(self, key: str) -> list[NodeInfo]:
        """[primary, replica 1, replica 2, ...]: `replication_factor` distinct nodes,
        walking clockwise around the ring from the key's position."""
        ids = self.ring.get_nodes(key, self.replication_factor)
        return [self._nodes.get(node_id) for node_id in ids]
