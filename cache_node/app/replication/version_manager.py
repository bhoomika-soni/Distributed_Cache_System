"""Remembers the last version of keys that no longer have a live entry.

Why this exists
---------------
Every key has a version that goes up by one on each write. Replicas use it to
decide whether an incoming replicated write is NEWER than what they hold
(apply it) or OLD/duplicate (ignore it). That makes replication idempotent and
safe against reordering - network retries can deliver messages late or twice.

The trap: DELETE removes the entry, and the version with it. Then

    SET k  -> v1,  DELETE k,  SET k  -> v1 again   (version restarted!)

and a replica that saw the delete would reject the new "v1" as old. The fix is
a TOMBSTONE: after a delete (or an LRU eviction) we keep "key k reached version
N" for a while, so the next SET continues from N+1. Replicas keep tombstones
too, so a late-arriving older SET cannot resurrect a deleted key.

Tombstones are bounded: they expire after `retention` seconds and there are at
most `max_entries` of them (oldest dropped first). After that window a key's
version history is forgotten - see "Limitations" in the README.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Callable


class VersionManager:
    def __init__(
        self,
        retention: float = 300.0,
        max_entries: int = 10_000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if retention <= 0 or max_entries < 1:
            raise ValueError("retention and max_entries must be positive")
        self._retention = retention
        self._max_entries = max_entries
        self._clock = clock
        # key -> (highest version seen, when this tombstone expires); oldest first
        self._floors: OrderedDict[str, tuple[int, float]] = OrderedDict()

    def record(self, key: str, version: int) -> None:
        """Remember that `key` reached `version` (it was deleted or evicted)."""
        previous = self._floors.get(key)
        if previous is not None:
            version = max(version, previous[0])
        self._floors[key] = (version, self._clock() + self._retention)
        self._floors.move_to_end(key)
        while len(self._floors) > self._max_entries:
            self._floors.popitem(last=False)  # drop the oldest tombstone

    def floor(self, key: str) -> int:
        """Highest remembered version for `key`, or 0 if none (or it expired)."""
        item = self._floors.get(key)
        if item is None:
            return 0
        version, expires_at = item
        if self._clock() >= expires_at:
            del self._floors[key]
            return 0
        return version

    def forget(self, key: str) -> None:
        """The key is live again, so its tombstone is no longer needed."""
        self._floors.pop(key, None)

    def purge(self) -> int:
        """Drop expired tombstones; return how many were removed."""
        now = self._clock()
        expired = [k for k, (_, expires_at) in self._floors.items() if now >= expires_at]
        for key in expired:
            del self._floors[key]
        return len(expired)

    def __len__(self) -> int:
        return len(self._floors)
