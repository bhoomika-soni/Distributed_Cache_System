"""The single-node cache engine: SET / GET / DELETE / EXISTS / TTL.

Design notes
------------
* Concurrency: every method here is plain synchronous code with no `await`.
  FastAPI runs on one asyncio event loop, and a coroutine can only be
  interrupted at an `await`, so each engine call runs start-to-finish without
  another request interleaving. That makes the engine safe without locks.
  (If we ever moved to threads, we would need a lock.)

* Expiration is two-layered:
    1. LAZY   - any operation that touches a key first checks whether it has
                expired and removes it if so (`_get_live`).
    2. ACTIVE - `purge_expired()` sweeps the whole store; a background task
                (ttl_manager.TTLCleaner) calls it periodically so keys that
                are never read again do not sit in memory forever.

* What counts as "using" a key for LRU purposes:
    GET and SET refresh recency.  EXISTS and TTL are metadata checks and do
    NOT refresh recency or count as cache hits/misses. (Redis makes a similar
    distinction for hit statistics.)

* Versions and replication (Phase 6):
    - The PRIMARY assigns versions: a new write is previous version + 1.
    - REPLICAS use `apply_set` / `apply_delete`, which store the primary's
      version and only accept strictly newer ones (so duplicates, retries and
      out-of-order delivery are harmless).
    - `VersionManager` keeps the last version of deleted/evicted keys, so
      version numbers never restart for a key whose history is still remembered.
    - Replicated writes are NOT counted as client SET operations in `stats`;
      they are tracked separately by the replication manager.

* The clock is injectable so tests can move time forward instantly instead of
  sleeping.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from app.cache.cache_entry import CacheEntry
from app.cache.lru_manager import LRUManager
from app.replication.version_manager import VersionManager

# Return values of ttl(), following the Redis convention.
TTL_MISSING = -2  # key does not exist (or has expired)
TTL_NO_EXPIRY = -1  # key exists and never expires

APPLIED = "applied"
STALE = "stale"  # replica already has this version or a newer one
EXPIRED = "expired"  # the replicated entry's expiry time has already passed


@dataclass
class CacheStats:
    """Raw counters. Phase 8 builds the metrics API on top of these."""

    sets: int = 0
    gets: int = 0
    deletes: int = 0
    hits: int = 0
    misses: int = 0
    expired: int = 0  # entries removed because their TTL ran out
    evictions: int = 0  # live entries removed to make room (LRU)


@dataclass(frozen=True)
class ApplyResult:
    applied: bool
    reason: str  # APPLIED | STALE | EXPIRED
    version: int  # the version this node holds for the key after the call


class CacheEngine:
    def __init__(
        self,
        capacity: int,
        clock: Callable[[], float] = time.time,
        tombstone_ttl: float = 300.0,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._clock = clock
        self._store = LRUManager()
        self.versions = VersionManager(
            retention=tombstone_ttl, max_entries=max(1000, capacity * 10), clock=clock
        )
        self.stats = CacheStats()

    # ------------------------------------------------------------------ SET
    def set(self, key: str, value: Any, ttl: float | None = None) -> CacheEntry:
        """Store or overwrite a key. `ttl` is in seconds; None means no expiry.

        Overwriting replaces the value AND the TTL (a SET without ttl removes
        any previous expiration) and bumps the version by one.
        """
        self._validate_key(key)
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be > 0 seconds (or None for no expiration)")

        now = self._clock()
        self.stats.sets += 1
        expires_at = None if ttl is None else now + ttl

        existing = self._get_live(key, now)
        if existing is not None:
            existing.value = value
            existing.last_accessed_at = now
            existing.expires_at = expires_at
            existing.version += 1
            self._store.touch(key)
            return existing

        # New key. If we remember a version for it (it was deleted or evicted
        # recently) continue from there instead of restarting at 1.
        version = self.versions.floor(key) + 1
        self.versions.forget(key)

        # Make room first if we are full.
        if len(self._store) >= self.capacity:
            self._evict_lru(now)

        entry = CacheEntry(
            key=key,
            value=value,
            created_at=now,
            last_accessed_at=now,
            expires_at=expires_at,
            version=version,
        )
        self._store.put(entry)
        return entry

    # ------------------------------------------------------------------ GET
    def get(self, key: str) -> CacheEntry | None:
        """Return the live entry, or None on a miss (absent or expired).

        Returns the entry object rather than just the value so callers can
        tell a miss apart from a stored `null`, and can read the version.
        Treat the returned entry as read-only.
        """
        now = self._clock()
        self.stats.gets += 1

        entry = self._get_live(key, now)
        if entry is None:
            self.stats.misses += 1
            return None

        self.stats.hits += 1
        entry.last_accessed_at = now
        self._store.touch(key)  # now the most recently used
        return entry

    # --------------------------------------------------------------- DELETE
    def delete(self, key: str) -> bool:
        """Remove a key. True if a live key was deleted, False if nothing was there."""
        return self.delete_with_version(key) is not None

    def delete_with_version(self, key: str) -> int | None:
        """Like delete(), but returns the version of the deletion (None if nothing
        was deleted). The primary replicates that version so replicas can tell a
        delete apart from an older SET that arrives late."""
        now = self._clock()
        self.stats.deletes += 1

        entry = self._store.remove(key)
        if entry is None:
            return None
        if entry.is_expired(now):
            self.stats.expired += 1  # it was already logically gone
            return None

        version = entry.version + 1
        self.versions.record(key, version)  # tombstone: next SET continues from here
        return version

    # --------------------------------------------------------------- EXISTS
    def exists(self, key: str) -> bool:
        return self._get_live(key, self._clock()) is not None

    # ------------------------------------------------------------------ TTL
    def ttl(self, key: str) -> float:
        """Seconds until expiry, TTL_NO_EXPIRY (-1), or TTL_MISSING (-2)."""
        now = self._clock()
        entry = self._get_live(key, now)
        if entry is None:
            return TTL_MISSING
        remaining = entry.ttl_remaining(now)
        return TTL_NO_EXPIRY if remaining is None else remaining

    # ------------------------------------------------- replica-side operations
    def apply_set(
        self,
        key: str,
        value: Any,
        version: int,
        created_at: float,
        expires_at: float | None,
    ) -> ApplyResult:
        """Apply a SET replicated from the primary.

        Stores the primary's version, creation time and ABSOLUTE expiry time.
        (If we received "ttl=60" instead, every hop would restart the clock
        and replicas would outlive the primary by the replication delay.)
        Only strictly newer versions are accepted.
        """
        now = self._clock()
        if expires_at is not None and now >= expires_at:
            return ApplyResult(False, EXPIRED, self._known_version(key, now))

        current = self._known_version(key, now)
        if version <= current:
            return ApplyResult(False, STALE, current)

        existing = self._store.get(key)  # live: _known_version just dropped it if expired
        if existing is not None:
            existing.value = value
            existing.created_at = created_at
            existing.last_accessed_at = now
            existing.expires_at = expires_at
            existing.version = version
            self._store.touch(key)
        else:
            if len(self._store) >= self.capacity:
                self._evict_lru(now)
            self._store.put(
                CacheEntry(
                    key=key,
                    value=value,
                    created_at=created_at,
                    last_accessed_at=now,
                    expires_at=expires_at,
                    version=version,
                )
            )
            self.versions.forget(key)
        return ApplyResult(True, APPLIED, version)

    def apply_delete(self, key: str, version: int) -> ApplyResult:
        """Apply a DELETE replicated from the primary (idempotent, version-checked).

        Leaves a tombstone so an older SET that arrives afterwards is rejected
        instead of resurrecting the key."""
        now = self._clock()
        current = self._known_version(key, now)
        if version <= current:
            return ApplyResult(False, STALE, current)
        self._store.remove(key)
        self.versions.record(key, version)
        return ApplyResult(True, APPLIED, version)

    # ---------------------------------------------------- background expiry
    def purge_expired(self) -> int:
        """Remove every expired entry; return how many were removed.

        This is a full O(n) scan. That is fine for a few thousand keys and
        keeps the code simple. (Alternatives: a min-heap ordered by expiry, or
        Redis-style random sampling - good "future improvements".)
        Also drops expired tombstones.
        """
        now = self._clock()
        removed = 0
        for entry in self._store.snapshot():  # snapshot: we remove while looping
            if entry.is_expired(now):
                self._store.remove(entry.key)
                removed += 1
        self.stats.expired += removed
        self.versions.purge()
        return removed

    def size(self) -> int:
        """Entries currently stored. May include expired ones not yet purged."""
        return len(self._store)

    # ------------------------------------------------------------- helpers
    def _get_live(self, key: str, now: float) -> CacheEntry | None:
        """Lazy expiration: return the entry if alive; drop it if expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.is_expired(now):
            self._store.remove(key)
            self.stats.expired += 1
            return None
        return entry

    def _known_version(self, key: str, now: float) -> int:
        """Newest version we know for `key`: its live entry, else its tombstone, else 0."""
        entry = self._get_live(key, now)
        return entry.version if entry is not None else self.versions.floor(key)

    def _evict_lru(self, now: float) -> None:
        victim = self._store.pop_lru()
        if victim is None:
            return
        # An expired victim was going to disappear anyway; don't report it as
        # a capacity eviction or the eviction metric would be misleading.
        if victim.is_expired(now):
            self.stats.expired += 1
        else:
            self.stats.evictions += 1
            # Remember its version so re-setting the key continues the numbering.
            self.versions.record(victim.key, victim.version)

    @staticmethod
    def _validate_key(key: str) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")
