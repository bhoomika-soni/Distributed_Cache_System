"""Re-syncs a node that came back (usually after a restart, with EMPTY memory).

What it does
------------
While a node was down, writes went to other nodes. When it returns it is
`recovering`: it takes no reads. This service asks every healthy peer for its
live entries (`/internal/cache/snapshot`), keeps the entries that this node
should hold (it is one of the key's `replication_factor` ring nodes), and sends
them to the node's replication endpoint. That endpoint is version-checked, so
re-sending is idempotent and an entry the node already has is just skipped.
When every source has been copied, the node becomes `healthy` and takes
traffic again. If anything fails the node stays `recovering` and the health
monitor starts another attempt on its next tick.

Known gaps (documented, not hidden)
-----------------------------------
* A key DELETED while the node was away is not removed from a node that did
  NOT lose its memory (e.g. a network partition rather than a restart).
  Fixing that needs tombstone transfer.
* Writes that arrive during the copy are replicated to the node by their
  primary (it is a valid replica target while recovering), but a write that
  slips between the snapshot and going live can still be missed.
* Snapshots are sent whole. Fine for thousands of keys, not for millions.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

from .node_client import NodeClient
from .node_manager import HEALTHY, RECOVERING, NodeInfo, NodeManager
from .node_router import NodeRouter

logger = logging.getLogger("gateway.recovery")


class ResyncAborted(Exception):
    """The node left the `recovering` state (e.g. it failed again) during the re-sync."""


class RecoveryService:
    def __init__(
        self,
        nodes: NodeManager,
        router: NodeRouter,
        client: NodeClient,
        concurrency: int = 10,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._nodes = nodes
        self._router = router
        self._client = client
        self._concurrency = concurrency
        self._clock = clock
        self._tasks: dict[str, asyncio.Task] = {}

    # -------------------------------------------------------------- lifecycle
    def is_running(self, node_id: str) -> bool:
        task = self._tasks.get(node_id)
        return task is not None and not task.done()

    def start(self, node_id: str) -> None:
        """Begin re-syncing `node_id` in the background (no-op if already running)."""
        if self.is_running(node_id):
            return
        task = asyncio.create_task(self._run(node_id), name=f"resync-{node_id}")
        self._tasks[node_id] = task
        task.add_done_callback(lambda t, nid=node_id: self._forget(nid, t))

    def _forget(self, node_id: str, task: asyncio.Task) -> None:
        if self._tasks.get(node_id) is task:
            del self._tasks[node_id]

    async def stop(self) -> None:
        """Cancel running re-syncs (graceful shutdown)."""
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def wait_idle(self) -> None:
        """Wait for running re-syncs to finish (used by tests)."""
        tasks = list(self._tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------ work
    async def _run(self, node_id: str) -> None:
        target = self._nodes.get(node_id)
        logger.info("re-syncing node %s", node_id)
        try:
            result = await self.resync(target)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            logger.warning("re-sync of %s failed (will retry): %s", node_id, reason)
            self._nodes.record_resync_failure(node_id, reason)
            return
        if self._nodes.mark_healthy(node_id, result):
            logger.info(
                "node %s re-synced (%d key(s) copied) and back in service", node_id, result["copied"]
            )

    async def resync(self, target: NodeInfo) -> dict:
        """Copy to `target` every entry it should hold from the healthy peers.

        Returns a summary; raises if a peer or the target fails mid-way.
        """
        sources = [
            n for n in self._nodes.all() if n.node_id != target.node_id and n.status == HEALTHY
        ]
        copied = already_current = 0
        for source in sources:
            if target.status != RECOVERING:
                raise ResyncAborted(f"{target.node_id} is no longer recovering")
            snapshot = await self._client.snapshot(source)
            # Keep only entries this node is a rightful holder of. get_nodes() ignores
            # health here on purpose: "who SHOULD hold this key", not "who is up".
            owned = [
                entry
                for entry in snapshot["entries"]
                if target.node_id
                in self._router.ring.get_nodes(entry["key"], self._router.replication_factor)
            ]
            for applied in await self._apply_all(target, owned):
                if applied:
                    copied += 1
                else:
                    already_current += 1
        return {
            "ok": True,
            "at": self._clock(),
            "sources": [s.node_id for s in sources],
            "copied": copied,
            "already_current": already_current,
        }

    async def _apply_all(self, target: NodeInfo, entries: list[dict]) -> list[bool]:
        semaphore = asyncio.Semaphore(self._concurrency)  # bounded parallelism

        async def send(entry: dict) -> bool:
            async with semaphore:
                result = await self._client.apply(
                    target,
                    {
                        "op": "set",
                        "key": entry["key"],
                        "value": entry["value"],
                        "version": entry["version"],
                        "created_at": entry["created_at"],
                        "expires_at": entry["expires_at"],  # absolute expiry, never a fresh TTL
                        "origin": "gateway-resync",
                        "sent_at": self._clock(),
                    },
                )
                return bool(result.get("applied"))

        return list(await asyncio.gather(*(send(e) for e in entries)))
