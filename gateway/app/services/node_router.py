"""Decides which nodes handle a key, using a consistent hash ring and node health.

The ring itself never changes when a node fails: it always answers "who SHOULD
own this key". Health only decides who is eligible right now, by passing the
set of usable nodes to `ring.get_nodes(..., available=...)`, which skips the
rest and walks on to the next node clockwise. When a node recovers, keys are
found where they were.
"""
from __future__ import annotations

from typing import Iterable

from .consistent_hash import HashRing
from .node_manager import NodeInfo, NodeManager


class NodeRouter:
    def __init__(self, nodes: NodeManager, virtual_nodes: int, replication_factor: int = 1) -> None:
        self._nodes = nodes
        self.replication_factor = replication_factor
        self.ring = HashRing(virtual_nodes)
        for node in nodes.all():
            self.ring.add_node(node.node_id)

    def route(self, key: str) -> NodeInfo:
        """The key's NORMAL primary, ignoring health. Used to tell when a request
        was served by a fallback node."""
        return self._nodes.get(self.ring.get_node(key))

    def read_candidates(self, key: str) -> list[NodeInfo]:
        """Nodes to try for a read, in order: the key's first `replication_factor`
        usable nodes. Unhealthy and recovering nodes are skipped."""
        ids = self.ring.get_nodes(key, self.replication_factor, available=self._nodes.readable_ids())
        return [self._nodes.get(i) for i in ids]

    def acting_primaries(self, key: str) -> list[NodeInfo]:
        """Nodes that may act as primary for a write, in order. Normally the ring
        primary first; if it is down, the next usable node takes over.
        Recovering nodes are excluded (their version numbers may be behind)."""
        ids = self.ring.get_nodes(key, len(self._nodes), available=self._nodes.readable_ids())
        return [self._nodes.get(i) for i in ids]

    def replicas_for(
        self, key: str, primary: NodeInfo, exclude: Iterable[str] = ()
    ) -> list[NodeInfo]:
        """Where `primary` should replicate a write: the next `replication_factor - 1`
        eligible nodes clockwise. Recovering nodes ARE eligible (they catch up);
        the primary itself and `exclude` (nodes that just failed) are not."""
        if self.replication_factor <= 1:
            return []
        eligible = self._nodes.replica_ids() - set(exclude) - {primary.node_id}
        ids = self.ring.get_nodes(key, self.replication_factor - 1, available=eligible)
        return [self._nodes.get(i) for i in ids]
