"""Background task that periodically removes expired keys."""
from __future__ import annotations

import asyncio
import logging

from app.cache.cache_engine import CacheEngine

logger = logging.getLogger(__name__)


class TTLCleaner:
    """Calls `engine.purge_expired()` every `interval` seconds.

    Lazy expiration alone is not enough: a key that is never read again would
    stay in memory forever. This task is the safety net.
    """

    def __init__(self, engine: CacheEngine, interval: float) -> None:
        if interval <= 0:
            raise ValueError("interval must be > 0")
        self._engine = engine
        self._interval = interval
        self._task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start the loop on the running event loop. Safe to call twice."""
        if self.is_running:
            return
        self._task = asyncio.create_task(self._run(), name="ttl-cleaner")

    async def stop(self) -> None:
        """Cancel the loop and wait for it to finish (graceful shutdown)."""
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            try:
                removed = self._engine.purge_expired()
                if removed:
                    logger.debug("TTL cleanup removed %d expired key(s)", removed)
            except Exception:
                # One bad pass must not kill the cleaner for good.
                logger.exception("TTL cleanup pass failed; retrying next interval")
