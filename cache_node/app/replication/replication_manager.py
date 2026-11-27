"""Primary -> replica replication: asynchronous, bounded retries, fully tracked.

How it works
------------
1. A client write reaches the PRIMARY (the gateway picked it with the hash ring).
2. The primary applies it locally and answers the client immediately.
3. `replicate()` then starts one background task per replica. Each task POSTs
   the write to the replica's /internal/replication/apply endpoint.

Consistency (be honest about it)
--------------------------------
This is ASYNCHRONOUS replication: the client's success response means "the
primary has it", NOT "the replicas have it". For a short time (the replication
lag) a replica can serve older data or none at all, and if the primary dies
inside that window the write can be lost. We do not claim strong consistency.
The model is eventual consistency: replicas converge once deliveries succeed,
because version numbers make every apply idempotent and order-independent.

Failures are never silent
-------------------------
A delivery is retried a bounded number of times (exponential backoff + jitter,
no retry storms). If it still fails, we (1) bump `failed`, (2) log an ERROR,
and (3) keep a record in a bounded list exposed by the stats endpoint. There is
no automatic repair yet: a replica that missed a write stays behind until the
next write to that key arrives - see "Limitations" in the README.

Loop prevention
---------------
Replicas apply writes with `engine.apply_*` and NEVER call `replicate()`. The
apply endpoint has no replica list and nothing there can start another
replication, so a write travels exactly one hop: primary -> replica.
`replicate()` also refuses to target the node itself.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import httpx

logger = logging.getLogger("replication")


@dataclass
class ReplicationStats:
    # sending side (this node acting as primary); one unit = one write to one replica
    scheduled: int = 0
    succeeded: int = 0
    failed: int = 0
    retries: int = 0
    # receiving side (this node acting as replica)
    received: int = 0
    applied: int = 0
    ignored_stale: int = 0
    ignored_expired: int = 0
    last_lag_seconds: float | None = None  # approximate: assumes roughly synced clocks


@dataclass
class ReplicationManager:
    node_id: str
    http: httpx.AsyncClient
    cluster_secret: str
    max_retries: int = 2
    backoff_base: float = 0.05
    max_failure_records: int = 100
    clock: Callable[[], float] = time.time
    stats: ReplicationStats = field(default_factory=ReplicationStats)

    def __post_init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()
        self.recent_failures: deque[dict] = deque(maxlen=self.max_failure_records)

    # ------------------------------------------------------------- sending
    def replicate(self, payload: dict[str, Any], targets: Iterable[Any]) -> list[str]:
        """Start background delivery of `payload` to each target; return immediately.

        `targets` are objects with `.node_id` and `.url`. Returns the node ids
        that were scheduled. Must be called from inside the running event loop.
        """
        scheduled: list[str] = []
        for target in targets:
            if target.node_id == self.node_id:
                logger.warning("refusing to replicate to myself (%s)", self.node_id)
                continue
            self.stats.scheduled += 1
            task = asyncio.create_task(self._deliver(target, payload))
            self._tasks.add(task)  # keep a reference, or the task could be garbage collected
            task.add_done_callback(self._tasks.discard)
            scheduled.append(target.node_id)
        return scheduled

    async def _deliver(self, target: Any, payload: dict[str, Any]) -> None:
        attempts = self.max_retries + 1
        error = "unknown"
        made = 0
        try:
            for attempt in range(attempts):
                made = attempt + 1
                try:
                    resp = await self.http.post(
                        f"{target.url}/internal/replication/apply",
                        json=payload,
                        headers={"X-Cluster-Secret": self.cluster_secret},
                    )
                except httpx.TransportError as exc:
                    error = f"{type(exc).__name__}: {exc}".rstrip(": ")
                else:
                    if resp.is_success:
                        # 200 even when the replica ignored it as stale: it HAS
                        # this write or a newer one, so delivery succeeded.
                        self.stats.succeeded += 1
                        return
                    error = f"HTTP {resp.status_code}"
                    if resp.status_code < 500:
                        break  # 4xx (e.g. wrong secret) will not fix itself: do not retry
                if attempt < attempts - 1:
                    self.stats.retries += 1
                    await asyncio.sleep(self._backoff(attempt))
        except asyncio.CancelledError:
            self._record_failure(target, payload, "cancelled during shutdown", made)
            raise
        self._record_failure(target, payload, error, made)

    def _backoff(self, attempt: int) -> float:
        return self.backoff_base * (2 ** attempt) + random.uniform(0, self.backoff_base)

    def _record_failure(self, target: Any, payload: dict[str, Any], error: str, attempts: int) -> None:
        self.stats.failed += 1
        record = {
            "op": payload["op"],
            "key": payload["key"],
            "version": payload["version"],
            "target": target.node_id,
            "error": error,
            "attempts": attempts,
            "at": self.clock(),
        }
        self.recent_failures.append(record)
        logger.error(
            "replication FAILED: %s key=%r v%s -> %s after %d attempt(s): %s",
            record["op"], record["key"], record["version"], target.node_id, attempts, error,
        )

    # ------------------------------------------------------------ receiving
    def record_received(self, reason: str, sent_at: float) -> None:
        """Called by the apply endpoint with the engine's verdict (applied/stale/expired)."""
        self.stats.received += 1
        if reason == "applied":
            self.stats.applied += 1
        elif reason == "stale":
            self.stats.ignored_stale += 1
        elif reason == "expired":
            self.stats.ignored_expired += 1
        self.stats.last_lag_seconds = max(0.0, self.clock() - sent_at)

    # ---------------------------------------------------------- lifecycle
    @property
    def in_flight(self) -> int:
        return len(self._tasks)

    async def drain(self, timeout: float | None = None) -> None:
        """Wait for pending deliveries (they finish or fail on their own)."""
        pending = set(self._tasks)
        if pending:
            await asyncio.wait(pending, timeout=timeout)

    async def close(self, timeout: float = 3.0) -> None:
        """Graceful shutdown: give in-flight deliveries a moment, then cancel the rest.
        Cancelled deliveries are recorded as failures, not silently dropped."""
        await self.drain(timeout)
        leftover = set(self._tasks)
        for task in leftover:
            task.cancel()
        if leftover:
            await asyncio.gather(*leftover, return_exceptions=True)

    def snapshot(self) -> dict:
        s = self.stats
        return {
            "node_id": self.node_id,
            "sent": {
                "scheduled": s.scheduled,
                "succeeded": s.succeeded,
                "failed": s.failed,
                "retries": s.retries,
                "in_flight": self.in_flight,
            },
            "received": {
                "total": s.received,
                "applied": s.applied,
                "ignored_stale": s.ignored_stale,
                "ignored_expired": s.ignored_expired,
                "last_lag_seconds": s.last_lag_seconds,
            },
            "recent_failures": list(self.recent_failures),
        }
