"""Registry of cache nodes plus their health state.

Node states
-----------
    unknown     never checked yet (just after gateway start). Treated as usable.
    healthy     answering health checks; takes reads and writes.
    unhealthy   failed `failure_threshold` times in a row; gets NO traffic.
    recovering  answered `recovery_threshold` probes in a row after being
                unhealthy. It may have restarted EMPTY or missed writes, so it
                gets NO reads and cannot be the primary for a write, but it does
                receive replicated writes while the recovery service re-syncs it.
                When the re-sync succeeds it becomes healthy.

    unknown/healthy --(N failures)--> unhealthy --(M good probes)--> recovering
    recovering --(re-sync ok)--> healthy        recovering --(any failure)--> unhealthy

Why thresholds?  One lost packet must not take a node out of service, and one
lucky probe must not put a flapping node straight back. Requiring N failures /
M successes in a row (hysteresis) keeps the status stable.

Why `recovering` exists: a restarted node has an empty memory. If it served
reads immediately, every key it should have would be a miss; if it were a
write primary, it would restart version numbers at 1 and its writes would be
rejected as stale by replicas. So it is re-synced from its peers first.

This module has no runtime dependency on the rest of the gateway.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # annotation only
    from app.config import NodeConfig

logger = logging.getLogger("gateway.nodes")

UNKNOWN = "unknown"
HEALTHY = "healthy"
UNHEALTHY = "unhealthy"
RECOVERING = "recovering"


@dataclass
class NodeInfo:
    node_id: str
    url: str
    status: str = UNKNOWN
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_checked_at: float | None = None  # last health probe (any outcome)
    last_success_at: float | None = None
    last_error: str | None = None
    last_status_change_at: float | None = None
    resync_attempts: int = 0
    last_resync: dict | None = None  # outcome of the latest re-sync


class NodeManager:
    """Static registry built from the NODES setting, with health state per node.

    Everything runs on the gateway's single event-loop thread, so no locks.
    """

    def __init__(
        self,
        configs: list[NodeConfig],
        failure_threshold: int = 2,
        recovery_threshold: int = 2,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._nodes: dict[str, NodeInfo] = {
            c.node_id: NodeInfo(node_id=c.node_id, url=c.url) for c in configs
        }
        self._failure_threshold = failure_threshold
        self._recovery_threshold = recovery_threshold
        self._clock = clock

    # -------------------------------------------------------------- lookups
    def all(self) -> list[NodeInfo]:
        return list(self._nodes.values())

    def get(self, node_id: str) -> NodeInfo | None:
        return self._nodes.get(node_id)

    def __len__(self) -> int:
        return len(self._nodes)

    def readable_ids(self) -> set[str]:
        """Nodes that may serve reads and act as write primary."""
        return {n.node_id for n in self._nodes.values() if n.status in (HEALTHY, UNKNOWN)}

    def replica_ids(self) -> set[str]:
        """Nodes that may RECEIVE replicated writes: the readable ones plus
        recovering nodes (so they catch up while being re-synced)."""
        return {
            n.node_id for n in self._nodes.values() if n.status in (HEALTHY, UNKNOWN, RECOVERING)
        }

    # ---------------------------------------------------------- transitions
    def record_success(self, node_id: str, probe: bool = False) -> None:
        """A request or health probe succeeded. `probe=True` for health checks."""
        node = self._nodes[node_id]
        now = self._clock()
        node.consecutive_failures = 0
        node.consecutive_successes += 1
        node.last_success_at = now
        if probe:
            node.last_checked_at = now

        if node.status == UNKNOWN:
            self._set_status(node, HEALTHY, "first successful check")
        elif (
            node.status == UNHEALTHY
            and probe  # only deliberate probes bring a node back, never stray request results
            and node.consecutive_successes >= self._recovery_threshold
        ):
            node.resync_attempts = 0
            self._set_status(node, RECOVERING, f"{node.consecutive_successes} good probes in a row")

    def record_failure(self, node_id: str, error: str, probe: bool = False) -> None:
        """A request or health probe failed because the node was unreachable/too slow."""
        node = self._nodes[node_id]
        node.consecutive_successes = 0
        node.consecutive_failures += 1
        node.last_error = error
        if probe:
            node.last_checked_at = self._clock()

        if node.status == RECOVERING:
            self._set_status(node, UNHEALTHY, f"failed while recovering: {error}")
        elif node.status in (UNKNOWN, HEALTHY) and node.consecutive_failures >= self._failure_threshold:
            self._set_status(
                node, UNHEALTHY, f"{node.consecutive_failures} failures in a row: {error}"
            )

    def mark_healthy(self, node_id: str, resync: dict) -> bool:
        """Re-sync finished: put a RECOVERING node back into service."""
        node = self._nodes[node_id]
        if node.status != RECOVERING:
            return False
        node.last_resync = resync
        node.last_error = None
        node.consecutive_failures = 0
        self._set_status(node, HEALTHY, "re-sync complete")
        return True

    def record_resync_failure(self, node_id: str, error: str) -> None:
        node = self._nodes[node_id]
        node.resync_attempts += 1
        node.last_resync = {
            "ok": False,
            "error": error,
            "attempt": node.resync_attempts,
            "at": self._clock(),
        }

    def _set_status(self, node: NodeInfo, status: str, reason: str) -> None:
        previous, node.status = node.status, status
        node.last_status_change_at = self._clock()
        level = logging.WARNING if status == UNHEALTHY else logging.INFO
        logger.log(level, "node %s: %s -> %s (%s)", node.node_id, previous, status, reason)
