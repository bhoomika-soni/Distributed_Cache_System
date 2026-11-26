"""A single cached item and its metadata."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    """One key/value pair plus the bookkeeping the cache needs.

    All timestamps are wall-clock epoch seconds (time.time()), NOT a monotonic
    clock. That is deliberate: in Phase 6 we replicate `expires_at` to other
    nodes, and a monotonic value from one machine means nothing on another.
    """

    key: str
    value: Any
    created_at: float
    last_accessed_at: float
    expires_at: float | None  # None means "never expires"
    version: int  # starts at 1, +1 on every overwrite

    def is_expired(self, now: float) -> bool:
        return self.expires_at is not None and now >= self.expires_at

    def ttl_remaining(self, now: float) -> float | None:
        """Seconds left, or None if the entry has no expiration."""
        if self.expires_at is None:
            return None
        return max(0.0, self.expires_at - now)
