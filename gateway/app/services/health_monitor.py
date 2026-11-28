"""Background task that probes every node and keeps their status up to date.

Every `interval` seconds all nodes are probed CONCURRENTLY (one slow node must
not delay the checks of the others). Each probe is a single attempt with the
normal request timeout; the failure/recovery thresholds in NodeManager decide
when a status actually changes.

Besides probing, each tick starts the re-sync of any node that has reached
`recovering` (see recovery_service). A re-sync that failed is simply started
again on the next tick - bounded by the interval, never a tight loop.
"""
from __future__ import annotations

import asyncio
import logging

from .node_client import NodeClient
from .node_manager import RECOVERING, NodeInfo, NodeManager
from .recovery_service import RecoveryService

logger = logging.getLogger("gateway.health")


class HealthMonitor:
    def __init__(
        self,
        nodes: NodeManager,
        client: NodeClient,
        recovery: RecoveryService,
        interval: float,
    ) -> None:
        if interval <= 0:
            raise ValueError("interval must be > 0")
        self._nodes = nodes
        self._client = client
        self._recovery = recovery
        self._interval = interval
        self._task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start the loop on the running event loop. Safe to call twice."""
        if self.is_running:
            return
        self._task = asyncio.create_task(self._run(), name="health-monitor")

    async def stop(self) -> None:
        """Cancel the loop and wait for it (graceful shutdown)."""
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def check_all(self) -> None:
        """One round: probe every node, then start re-syncs where needed."""
        nodes = self._nodes.all()
        await asyncio.gather(*(self._probe(node) for node in nodes))
        for node in nodes:
            if node.status == RECOVERING:
                self._recovery.start(node.node_id)

    async def _probe(self, node: NodeInfo) -> None:
        try:
            await self._client.probe(node)
        except Exception as exc:  # any failure to get a healthy answer counts against the node
            self._nodes.record_failure(node.node_id, str(exc) or type(exc).__name__, probe=True)
        else:
            self._nodes.record_success(node.node_id, probe=True)

    async def _run(self) -> None:
        while True:
            try:
                await self.check_all()
            except Exception:
                # One bad round must not kill monitoring for good.
                logger.exception("health check round failed; retrying next interval")
            await asyncio.sleep(self._interval)
