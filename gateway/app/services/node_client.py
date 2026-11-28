"""HTTP client for gateway -> cache node calls.

Self-contained on purpose: it imports nothing from the gateway's `app` package
at runtime, so it can be unit-tested in isolation.

Timeouts and retries (the rules, in one place)
----------------------------------------------
* Every request has a timeout (set on the shared httpx client).
* We retry at most `max_retries` extra times, with exponential backoff plus
  jitter, so a struggling node is not hit by a burst of instant retries
  (a "retry storm").
* We only retry when it is SAFE:
    - the request never reached the node (connection refused / connect
      timeout / pool timeout)  -> safe for every operation
    - the request may have reached the node (read timeout, connection reset)
      -> only retried for idempotent operations (get, exists, ttl, delete).
        SET is NOT retried here: the first attempt may already have been
        applied, and applying it twice would bump the version twice.
* We never retry on an HTTP response. A 404 is a normal cache miss; a 401 or
  5xx will not be fixed by asking again.
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING, Any, Sequence
from urllib.parse import quote

import httpx

if TYPE_CHECKING:  # annotation only; avoids a runtime import of the gateway package
    from app.services.node_manager import NodeInfo

logger = logging.getLogger("gateway.node_client")


class NodeError(Exception):
    def __init__(self, node_id: str, message: str) -> None:
        super().__init__(message)
        self.node_id = node_id


class NodeUnavailableError(NodeError):
    """Could not connect to the node, or the connection broke."""


class NodeTimeoutError(NodeUnavailableError):
    """The node did not answer in time."""


class NodeResponseError(NodeError):
    """The node answered, but not with what we expect (401, 5xx, bad JSON...)."""

    def __init__(self, node_id: str, status_code: int, message: str) -> None:
        super().__init__(node_id, message)
        self.status_code = status_code


class NodeClient:
    def __init__(
        self,
        http: httpx.AsyncClient,
        cluster_secret: str,
        max_retries: int,
        backoff_base: float = 0.05,
    ) -> None:
        self._http = http
        self._headers = {"X-Cluster-Secret": cluster_secret}
        self._max_retries = max_retries
        self._backoff_base = backoff_base

    # ------------------------------------------------------------ operations
    async def set(
        self, node: NodeInfo, key: str, value: Any, ttl: float | None,
        replicas: Sequence[NodeInfo] = (),
    ) -> dict:
        """`node` is the primary; it replicates to `replicas` in the background."""
        resp = await self._send(
            node, "POST", "/internal/cache/set", idempotent=False,
            json={"key": key, "value": value, "ttl": ttl, "replicas": _targets(replicas)},
        )
        return self._ok_json(node, resp)

    async def get(self, node: NodeInfo, key: str) -> dict | None:
        """Returns None on a cache miss (node answered 404)."""
        resp = await self._send(node, "GET", f"/internal/cache/get/{_enc(key)}", idempotent=True)
        if resp.status_code == 404:
            return None
        return self._ok_json(node, resp)

    async def delete(self, node: NodeInfo, key: str, replicas: Sequence[NodeInfo] = ()) -> dict:
        body = {"replicas": _targets(replicas)} if replicas else None
        resp = await self._send(
            node, "DELETE", f"/internal/cache/delete/{_enc(key)}", idempotent=True, json=body
        )
        return self._ok_json(node, resp)

    async def exists(self, node: NodeInfo, key: str) -> dict:
        resp = await self._send(node, "GET", f"/internal/cache/exists/{_enc(key)}", idempotent=True)
        return self._ok_json(node, resp)

    async def ttl(self, node: NodeInfo, key: str) -> dict:
        resp = await self._send(node, "GET", f"/internal/cache/ttl/{_enc(key)}", idempotent=True)
        return self._ok_json(node, resp)

    # ------------------------------------------ health checks and recovery
    async def probe(self, node: NodeInfo) -> dict:
        """One health check: a single attempt, no retries. The health monitor's
        thresholds (N failures in a row) ARE the retry policy here."""
        try:
            resp = await self._http.get(node.url + "/internal/health", headers=self._headers)
        except httpx.TransportError as exc:
            self._raise_unavailable(node, exc, 1)
        return self._ok_json(node, resp)

    async def snapshot(self, node: NodeInfo) -> dict:
        """Every live entry of a node (used to re-sync a restarted node)."""
        resp = await self._send(node, "GET", "/internal/cache/snapshot", idempotent=True)
        return self._ok_json(node, resp)

    async def apply(self, node: NodeInfo, payload: dict) -> dict:
        """Send one write to a node's replication endpoint. Safe to retry: the node
        applies it only if the version is newer than what it holds."""
        resp = await self._send(
            node, "POST", "/internal/replication/apply", idempotent=True, json=payload
        )
        return self._ok_json(node, resp)

    # --------------------------------------------------------------- plumbing
    async def _send(
        self, node: NodeInfo, method: str, path: str, *, idempotent: bool, json: Any = None
    ) -> httpx.Response:
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            try:
                return await self._http.request(
                    method, node.url + path, json=json, headers=self._headers
                )
            except httpx.TransportError as exc:
                never_sent = isinstance(
                    exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
                )
                is_last = attempt == attempts - 1
                if is_last or not (never_sent or idempotent):
                    self._raise_unavailable(node, exc, attempt + 1)
                logger.warning(
                    "%s %s on %s failed (%s), retrying (attempt %d of %d)",
                    method, path, node.node_id, type(exc).__name__, attempt + 1, attempts,
                )
                await asyncio.sleep(self._backoff(attempt))
        raise AssertionError("unreachable")  # the loop always returns or raises

    def _backoff(self, attempt: int) -> float:
        base = self._backoff_base
        return base * (2 ** attempt) + random.uniform(0, base)

    @staticmethod
    def _raise_unavailable(node: NodeInfo, exc: httpx.TransportError, attempts: int) -> None:
        reason = f"{type(exc).__name__} after {attempts} attempt(s)"
        # 504 (timeout) only when we reached the node and it was too slow to answer.
        # A connect timeout means we never reached it - on Windows a closed port
        # shows up as a connect timeout instead of "connection refused" - and a
        # pool timeout is the gateway's own overload. Both are 503 "unavailable".
        if isinstance(exc, httpx.TimeoutException) and not isinstance(
            exc, (httpx.ConnectTimeout, httpx.PoolTimeout)
        ):
            raise NodeTimeoutError(node.node_id, f"Node {node.node_id} timed out ({reason})") from exc
        raise NodeUnavailableError(node.node_id, f"Node {node.node_id} unavailable ({reason})") from exc

    @staticmethod
    def _ok_json(node: NodeInfo, resp: httpx.Response) -> dict:
        if not resp.is_success:
            raise NodeResponseError(
                node.node_id, resp.status_code,
                f"Node {node.node_id} returned HTTP {resp.status_code}",
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise NodeResponseError(
                node.node_id, resp.status_code, f"Node {node.node_id} returned invalid JSON"
            ) from exc


def _targets(replicas: Sequence[NodeInfo]) -> list[dict]:
    """Replica list in the wire format the node's SetRequest/DeleteRequest expects."""
    return [{"node_id": r.node_id, "url": r.url} for r in replicas]


def _enc(key: str) -> str:
    """Percent-encode a key for use in a URL path (':' '?' '#' '%' spaces, unicode...)."""
    return quote(key, safe="")
