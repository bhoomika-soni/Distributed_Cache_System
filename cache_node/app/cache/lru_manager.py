"""Storage that remembers recency order, so we can evict the least recently used.

How it works
------------
Python's OrderedDict is a hash table (O(1) lookup) combined with a doubly
linked list that records insertion order (O(1) reordering). We use the order
as a recency queue:

    first item  -> least recently used (LRU)  <- eviction candidate
    ...
    last item   -> most recently used (MRU)

    touch(key)  = move_to_end(key)   O(1)  "this key was just used"
    pop_lru()   = popitem(last=False) O(1)  "drop the oldest-used key"

Note this class only tracks ORDER. Deciding *when* to evict (capacity) and what
counts as "used" is the cache engine's job.
"""
from __future__ import annotations

from collections import OrderedDict

from app.cache.cache_entry import CacheEntry


class LRUManager:
    def __init__(self) -> None:
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, key: str) -> CacheEntry | None:
        """Look up an entry WITHOUT changing its recency."""
        return self._entries.get(key)

    def touch(self, key: str) -> None:
        """Mark an existing key as most recently used."""
        self._entries.move_to_end(key)

    def put(self, entry: CacheEntry) -> None:
        """Insert or replace an entry and mark it most recently used."""
        self._entries[entry.key] = entry
        self._entries.move_to_end(entry.key)

    def remove(self, key: str) -> CacheEntry | None:
        return self._entries.pop(key, None)

    def pop_lru(self) -> CacheEntry | None:
        """Remove and return the least recently used entry (None if empty)."""
        if not self._entries:
            return None
        _, entry = self._entries.popitem(last=False)
        return entry

    def snapshot(self) -> list[CacheEntry]:
        """A copy of all entries, safe to iterate while we remove from the cache."""
        return list(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)
