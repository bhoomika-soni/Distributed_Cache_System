"""Gateway business logic: pick nodes, call them, and survive node failures.

Failure rules (the heart of fault tolerance)
--------------------------------------------
READS (get / exists / ttl) try the key's usable nodes in ring order:
  * node unreachable OR too slow -> count it against that node, try the next one.
    (A replica may hold slightly older data than the primary: asynchronous
    replication means fallback reads are eventually consistent.)
  * node answers "miss" -> that IS the answer. We do NOT ask other nodes: that
    would multiply the cost of every genuine miss by the replication factor.
  * every candidate failed -> re-raise the last error (503 unreachable / 504 slow).

WRITES (set / delete) go to an "acting primary" plus replicas:
  * primary UNREACHABLE (request never arrived) -> safe to try the next node.
  * primary TOO SLOW (it may have applied the write) -> do NOT write elsewhere;
    the outcome is unknown, so answer 504. Writing it to a second node could
    create two diverging copies. The client can retry.
  * nobody usable -> 503.

Every outcome also feeds the node's health record, so real traffic detects
failures faster than the periodic health check alone.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from .node_client import NodeClient, NodeError, NodeTimeoutError, NodeUnavailableError
from .node_manager import NodeInfo, NodeManager
from .node_router import NodeRouter


class NoAvailableNodesError(Exception):
    """No node is usable for the key right now (all unhealthy or recovering)."""


def _replication_fields(replicas: list[NodeInfo]) -> dict:
    return {
        "replicas": [r.node_id for r in replicas],
        # "async": the primary has the write when we answer; replicas get it
        # a moment later. "none": no replicas (replication factor 1, or none usable).
        "replication": "async" if replicas else "none",
    }


class CacheService:
    def __init__(self, router: NodeRouter, client: NodeClient, nodes: NodeManager) -> None:
        self._router = router
        self._client = client
        self._nodes = nodes

    # ------------------------------------------------------------------ reads
    async def get(self, key: str) -> dict | None:
        node, data, fallback = await self._read(key, lambda n: self._client.get(n, key))
        if data is None:
            return None
        return {**data, "node_id": node.node_id, "fallback": fallback}

    async def exists(self, key: str) -> dict:
        node, data, fallback = await self._read(key, lambda n: self._client.exists(n, key))
        return {**data, "node_id": node.node_id, "fallback": fallback}

    async def ttl(self, key: str) -> dict:
        node, data, fallback = await self._read(key, lambda n: self._client.ttl(n, key))
        return {**data, "node_id": node.node_id, "fallback": fallback}

    async def _read(
        self, key: str, call: Callable[[NodeInfo], Awaitable[Any]]
    ) -> tuple[NodeInfo, Any, bool]:
        candidates = self._router.read_candidates(key)
        if not candidates:
            raise NoAvailableNodesError(self._nothing_usable(key))
        normal_primary = self._router.route(key).node_id

        last_error: NodeError | None = None
        for node in candidates:
            try:
                result = await call(node)
            except NodeUnavailableError as exc:  # includes NodeTimeoutError
                self._nodes.record_failure(node.node_id, str(exc))
                last_error = exc
                continue  # fall back to the next replica
            self._nodes.record_success(node.node_id)
            return node, result, node.node_id != normal_primary
        raise last_error  # every candidate failed

    # ----------------------------------------------------------------- writes
    async def set(self, key: str, value: Any, ttl: float | None) -> dict:
        return await self._write(key, lambda primary, replicas: self._client.set(primary, key, value, ttl, replicas))

    async def delete(self, key: str) -> dict:
        return await self._write(key, lambda primary, replicas: self._client.delete(primary, key, replicas))

    async def _write(
        self, key: str, call: Callable[[NodeInfo, list[NodeInfo]], Awaitable[dict]]
    ) -> dict:
        primaries = self._router.acting_primaries(key)
        if not primaries:
            raise NoAvailableNodesError(self._nothing_usable(key))
        normal_primary = self._router.route(key).node_id

        failed: set[str] = set()
        last_error: NodeError | None = None
        for primary in primaries:
            replicas = self._router.replicas_for(key, primary, exclude=failed)
            try:
                data = await call(primary, replicas)
            except NodeTimeoutError as exc:
                # Slow, not necessarily down, and the write may have been applied.
                # Do not try another node: report the unknown outcome.
                self._nodes.record_failure(primary.node_id, str(exc))
                raise
            except NodeUnavailableError as exc:
                # Never reached the node, so it is safe to use the next one.
                self._nodes.record_failure(primary.node_id, str(exc))
                failed.add(primary.node_id)
                last_error = exc
                continue
            self._nodes.record_success(primary.node_id)
            return {
                **data,
                "node_id": primary.node_id,
                "fallback": primary.node_id != normal_primary,
                **_replication_fields(replicas),
            }
        raise last_error  # every candidate was unreachable

    def _nothing_usable(self, key: str) -> str:
        states = ", ".join(f"{n.node_id}={n.status}" for n in self._nodes.all())
        return f"No available node can serve key '{key}' (node status: {states})"
