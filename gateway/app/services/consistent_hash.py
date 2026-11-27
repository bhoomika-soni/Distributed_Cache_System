"""Consistent hashing, implemented from scratch.

The problem
-----------
We have N cache nodes and must decide which node owns each key. The obvious
answer is `hash(key) % N`, but it falls apart when N changes: add one node to a
3-node cluster and about 75% of all keys suddenly map to a different node. Every
one of those keys is now a cache miss, which stampedes the database behind the
cache. We want a scheme where adding or removing a node moves as few keys as
possible.

The idea: a hash ring
---------------------
Picture the hash space 0 .. 2**64-1 bent into a circle (the "ring").

  1. Hash every NODE onto the ring.
  2. Hash every KEY onto the same ring.
  3. A key belongs to the first node you meet walking CLOCKWISE from the key's
     position (wrapping around from the top of the ring back to 0).

          node-A
        .'      '.        key K1 -> walks clockwise -> node-B
       /          \\       key K2 -> walks clockwise -> node-A
  node-C            node-B

Adding a node only takes over keys between it and the previous node on the
ring, so only ~1/N of the keys move - and they all move TO the new node.
Removing a node hands only that node's keys to its clockwise neighbours.

Virtual nodes
-------------
With one point per node, the arcs between points are very uneven, so some nodes
get far more keys than others. Instead each physical node is placed on the ring
many times ("virtual nodes", e.g. 100 points named "node-1#0", "node-1#1", ...).
More points means smoother arcs, so load evens out, and a removed node's keys
are spread across ALL the other nodes instead of dumping onto one neighbour.

Why hashlib and not Python's built-in hash()
--------------------------------------------
hash() of a string is randomized per process (PYTHONHASHSEED). Two gateway
processes - or one gateway before and after a restart - would disagree on where
keys live. MD5 is deterministic everywhere. (We use it for even spread, not for
security, so MD5's weaknesses do not matter here.)

Complexity: lookup is O(log V) by binary search over V = nodes * virtual_nodes
ring points. Adding/removing a node rebuilds the sorted ring: O(V log V). That
is fine because membership changes are rare.
"""
from __future__ import annotations

import hashlib
from bisect import bisect_right


def hash_key(value: str) -> int:
    """Map a string to a position on the ring: the first 8 bytes of its MD5, as an int."""
    digest = hashlib.md5(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class HashRing:
    def __init__(self, virtual_nodes: int = 100) -> None:
        if virtual_nodes < 1:
            raise ValueError("virtual_nodes must be >= 1")
        self.virtual_nodes = virtual_nodes
        self._ring: list[tuple[int, str]] = []  # (position, node_id), sorted by position
        self._positions: list[int] = []  # just the positions, for bisect
        self._nodes: set[str] = set()

    # ------------------------------------------------------------ membership
    def add_node(self, node_id: str) -> None:
        if node_id in self._nodes:
            raise ValueError(f"node '{node_id}' is already on the ring")
        self._nodes.add(node_id)
        for i in range(self.virtual_nodes):
            self._ring.append((hash_key(f"{node_id}#{i}"), node_id))
        self._rebuild()

    def remove_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            raise KeyError(node_id)
        self._nodes.remove(node_id)
        self._ring = [point for point in self._ring if point[1] != node_id]
        self._rebuild()

    def _rebuild(self) -> None:
        # Sorting (position, node_id) tuples also makes the order deterministic
        # in the astronomically unlikely case of two points sharing a position.
        self._ring.sort()
        self._positions = [position for position, _ in self._ring]

    @property
    def nodes(self) -> list[str]:
        return sorted(self._nodes)

    # --------------------------------------------------------------- lookups
    def get_node(self, key: str) -> str:
        """The node that owns `key`: first ring point clockwise from the key's hash."""
        if not self._ring:
            raise LookupError("the hash ring has no nodes")
        # bisect_right finds the first point strictly greater than the key's
        # position; the modulo wraps past the last point back to the first.
        index = bisect_right(self._positions, hash_key(key)) % len(self._ring)
        return self._ring[index][1]

    def get_nodes(self, key: str, count: int, available: set[str] | None = None) -> list[str]:
        """Up to `count` DISTINCT nodes for `key`, in clockwise order.

        The first entry is the primary; the rest are the next distinct nodes
        around the ring - these become the replicas (Phase 6).

        `available` makes this "find the next available node": nodes not in the
        set are skipped. This is equivalent to removing the unavailable nodes
        from the ring, but without changing the ring itself, so when a node
        comes back its keys are found where they were.
        """
        if not self._ring:
            raise LookupError("the hash ring has no nodes")
        if count < 1:
            raise ValueError("count must be >= 1")

        start = bisect_right(self._positions, hash_key(key)) % len(self._ring)
        chosen: list[str] = []
        # One full lap around the ring at most; stop early once we have enough.
        for step in range(len(self._ring)):
            node_id = self._ring[(start + step) % len(self._ring)][1]
            if node_id in chosen:
                continue
            if available is not None and node_id not in available:
                continue
            chosen.append(node_id)
            if len(chosen) == count:
                break
        return chosen

    def __len__(self) -> int:
        """Number of points on the ring (nodes * virtual_nodes)."""
        return len(self._ring)
