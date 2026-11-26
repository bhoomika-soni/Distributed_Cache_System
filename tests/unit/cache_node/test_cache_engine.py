import pytest

from app.cache.cache_engine import TTL_MISSING, TTL_NO_EXPIRY, CacheEngine


# ----------------------------------------------------------- basic operations
def test_set_then_get_returns_value_and_metadata(cache, clock):
    cache.set("user:1", {"name": "John"})
    entry = cache.get("user:1")
    assert entry.value == {"name": "John"}
    assert entry.version == 1
    assert entry.created_at == clock.now
    assert entry.expires_at is None


def test_overwrite_bumps_version_and_keeps_created_at(cache, clock):
    cache.set("k", "v1")
    created = clock.now
    clock.advance(5)
    entry = cache.set("k", "v2")
    assert entry.value == "v2"
    assert entry.version == 2
    assert entry.created_at == created
    assert cache.size() == 1


def test_stored_null_is_distinguishable_from_a_miss(cache):
    cache.set("k", None)
    entry = cache.get("k")
    assert entry is not None and entry.value is None
    assert cache.get("other") is None


def test_delete(cache):
    cache.set("k", "v")
    assert cache.delete("k") is True
    assert cache.get("k") is None
    assert cache.delete("k") is False  # already gone


def test_exists_is_a_metadata_check_only(clock):
    """EXISTS must not count as a hit/miss nor refresh LRU recency."""
    cache = CacheEngine(capacity=2, clock=clock)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.exists("a") is True
    assert cache.exists("zzz") is False
    assert (cache.stats.gets, cache.stats.hits, cache.stats.misses) == (0, 0, 0)

    cache.set("c", 3)  # "a" is still the LRU key, so it is the one evicted
    assert cache.exists("a") is False


def test_stats_count_real_operations(cache):
    cache.set("a", 1)
    cache.set("a", 2)
    cache.get("a")  # hit
    cache.get("nope")  # miss
    cache.delete("a")
    s = cache.stats
    assert (s.sets, s.gets, s.hits, s.misses, s.deletes) == (2, 2, 1, 1, 1)


@pytest.mark.parametrize(
    "key, ttl",
    [("", None), (123, None), ("k", 0), ("k", -5)],
)
def test_invalid_input_is_rejected(cache, key, ttl):
    with pytest.raises(ValueError):
        cache.set(key, "v", ttl=ttl)


def test_capacity_must_be_positive(clock):
    with pytest.raises(ValueError):
        CacheEngine(capacity=0, clock=clock)


# ------------------------------------------------------------------------ TTL
def test_ttl_values(cache, clock):
    assert cache.ttl("missing") == TTL_MISSING
    cache.set("forever", 1)
    assert cache.ttl("forever") == TTL_NO_EXPIRY

    cache.set("temp", 1, ttl=10)
    clock.advance(4)
    assert cache.ttl("temp") == pytest.approx(6)
    clock.advance(6)  # exactly at expiry counts as expired
    assert cache.ttl("temp") == TTL_MISSING


def test_set_without_ttl_removes_previous_expiration(cache, clock):
    cache.set("k", "v", ttl=5)
    cache.set("k", "v2")
    clock.advance(100)
    assert cache.get("k").value == "v2"


def test_get_lazily_expires_entry(cache, clock):
    cache.set("k", "v", ttl=5)
    clock.advance(4.9)
    assert cache.get("k").value == "v"
    clock.advance(0.2)
    assert cache.get("k") is None
    assert cache.size() == 0  # physically removed, not just hidden
    assert cache.stats.expired == 1
    assert cache.stats.misses == 1


@pytest.mark.parametrize(
    "op, expected",
    [("exists", False), ("ttl", TTL_MISSING), ("delete", False)],
)
def test_every_operation_expires_lazily(cache, clock, op, expected):
    cache.set("k", "v", ttl=1)
    clock.advance(2)
    assert getattr(cache, op)("k") == expected
    assert cache.size() == 0
    assert cache.stats.expired == 1  # counted exactly once


def test_set_on_expired_key_acts_as_new_key(cache, clock):
    cache.set("k", "old", ttl=1)
    clock.advance(2)
    cache.set("k", "new")
    assert cache.get("k").value == "new"
    assert cache.size() == 1
    assert cache.stats.expired == 1


def test_purge_expired_removes_only_expired_entries(cache, clock):
    cache.set("short", 1, ttl=1)
    cache.set("long", 2, ttl=100)
    cache.set("forever", 3)
    clock.advance(2)

    assert cache.purge_expired() == 1
    assert cache.size() == 2
    assert cache.stats.expired == 1
    assert cache.purge_expired() == 0  # nothing left to purge


# ------------------------------------------------------------- capacity & LRU
def test_size_never_exceeds_capacity(cache):  # capacity = 3
    for i in range(10):
        cache.set(f"k{i}", i)
        assert cache.size() <= 3
    assert cache.size() == 3
    assert cache.stats.evictions == 7
    assert all(cache.exists(f"k{i}") for i in (7, 8, 9))  # newest survive


def test_updating_existing_key_at_capacity_does_not_evict(cache):
    for k in "abc":
        cache.set(k, 1)
    cache.set("a", 2)
    assert cache.size() == 3
    assert cache.stats.evictions == 0


def test_lru_evicts_the_least_recently_used_key(cache):
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    cache.get("a")  # "a" becomes most recent, so "b" is now the LRU key
    cache.set("d", 4)

    assert cache.exists("b") is False
    assert all(cache.exists(k) for k in ("a", "c", "d"))
    assert cache.stats.evictions == 1


def test_overwriting_a_key_refreshes_its_recency(cache):
    for k in "abc":
        cache.set(k, 1)
    cache.set("a", 99)  # refreshes "a"; "b" is now the LRU key
    cache.set("d", 4)
    assert cache.exists("b") is False
    assert cache.exists("a") is True


def test_ttl_check_does_not_refresh_recency(clock):
    cache = CacheEngine(capacity=2, clock=clock)
    cache.set("a", 1, ttl=100)
    cache.set("b", 2)
    cache.ttl("a")  # must NOT save "a" from eviction
    cache.set("c", 3)
    assert cache.exists("a") is False


def test_expired_eviction_victim_is_not_counted_as_eviction(clock):
    cache = CacheEngine(capacity=2, clock=clock)
    cache.set("a", 1, ttl=1)
    cache.set("b", 2)
    clock.advance(2)  # "a" is expired but still physically stored, and is LRU
    cache.set("c", 3)

    assert cache.stats.evictions == 0
    assert cache.stats.expired == 1
    assert cache.exists("b") and cache.exists("c")
