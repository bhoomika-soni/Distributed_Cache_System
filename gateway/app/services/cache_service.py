"""Gateway business logic: pick the responsible nodes, call the primary, add node info.

Routes stay thin (validation + HTTP status codes); this class decides WHAT
happens.

Writes (SET / DELETE) go to the PRIMARY together with the list of replicas; the
primary applies the write, answers, and replicates asynchronously. Reads go to
the primary only for now - replica fallback arrives in Phase 7.
"""
from __future__ import annotations

from typing import Any

from app.services.node_client import NodeClient
from app.services.node_router import NodeRouter


def _replication_fields(replicas: list) -> dict:
    return {
        "replicas": [r.node_id for r in replicas],
        # "async": the primary has the write when we answer; replicas get it
        # a moment later. "none" means replication factor 1 (no replicas).
        "replication": "async" if replicas else "none",
    }


class CacheService:
    def __init__(self, router: NodeRouter, client: NodeClient) -> None:
        self._router = router
        self._client = client

    async def set(self, key: str, value: Any, ttl: float | None) -> dict:
        primary, *replicas = self._router.replica_set(key)
        data = await self._client.set(primary, key, value, ttl, replicas)
        return {**data, "node_id": primary.node_id, **_replication_fields(replicas)}

    async def get(self, key: str) -> dict | None:
        node = self._router.route(key)
        data = await self._client.get(node, key)
        return None if data is None else {**data, "node_id": node.node_id}

    async def delete(self, key: str) -> dict:
        primary, *replicas = self._router.replica_set(key)
        data = await self._client.delete(primary, key, replicas)
        return {**data, "node_id": primary.node_id, **_replication_fields(replicas)}

    async def exists(self, key: str) -> dict:
        node = self._router.route(key)
        return {**await self._client.exists(node, key), "node_id": node.node_id}

    async def ttl(self, key: str) -> dict:
        node = self._router.route(key)
        return {**await self._client.ttl(node, key), "node_id": node.node_id}
