"""Gateway business logic: pick the responsible node, call it, add node info.

Routes stay thin (validation + HTTP status codes); this class decides WHAT
happens. In later phases replication and replica fallback are added here.
"""
from __future__ import annotations

from typing import Any

from app.services.node_client import NodeClient
from app.services.node_router import NodeRouter


class CacheService:
    def __init__(self, router: NodeRouter, client: NodeClient) -> None:
        self._router = router
        self._client = client

    async def set(self, key: str, value: Any, ttl: float | None) -> dict:
        node = self._router.route(key)
        data = await self._client.set(node, key, value, ttl)
        return {**data, "node_id": node.node_id}

    async def get(self, key: str) -> dict | None:
        node = self._router.route(key)
        data = await self._client.get(node, key)
        return None if data is None else {**data, "node_id": node.node_id}

    async def delete(self, key: str) -> dict:
        node = self._router.route(key)
        return {**await self._client.delete(node, key), "node_id": node.node_id}

    async def exists(self, key: str) -> dict:
        node = self._router.route(key)
        return {**await self._client.exists(node, key), "node_id": node.node_id}

    async def ttl(self, key: str) -> dict:
        node = self._router.route(key)
        return {**await self._client.ttl(node, key), "node_id": node.node_id}
