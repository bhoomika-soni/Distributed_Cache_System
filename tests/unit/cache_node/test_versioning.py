"""Versions, tombstones, and the replica-side apply operations."""
import pytest

from app.cache.cache_engine import APPLIED, EXPIRED, STALE, CacheEngine
from app.replication.version_manager import VersionManager


# ------------------------------------------------- version continuity (primary)
def test_versions_increment_on_each_overwrite(cache):
    assert [cache.set("k", i).version for i in range(3)] == [1, 2, 3]


def test_version_continues_after_delete_instead_of_restarting(cache):
    """Regression for the resurrection bug: SET v1, DELETE, SET must NOT be v1 again,
    or a replica that saw the delete would reject the new write as old."""
    cache.set("k", "a")
    assert cache.delete_with_version("k") == 2  # the delete is itself a versioned event
    assert cache.set("k", "b").version == 3


def test_delete_of_missing_key_returns_none_and_leaves_no_tombstone(cache):
    assert cache.delete_with_version("ghost") is None
    assert len(cache.versions) == 0
    assert cache.set("ghost", 1).version == 1


def test_plain_delete_still_returns_a_bool(cache):
    cache.set("k", 1)
    assert cache.delete("k") is True
    assert cache.delete("k") is False


def test_version_continues_after_lru_eviction(clock):
    cache = CacheEngine(capacity=2, clock=clock)
    cache.set("a", 1)
    cache.set("a", 2)  # a is at version 2
    cache.set("b", 1)
    cache.set("c", 1)  # evicts "a" (least recently used)
    assert not cache.exists("a")
    assert cache.set("a", 3).version == 3  # numbering continues


def test_tombstones_are_forgotten_after_the_retention_window(clock):
    cache = CacheEngine(capacity=5, clock=clock, tombstone_ttl=60)
    cache.set("k", 1)
    cache.delete("k")
    clock.advance(61)
    assert cache.set("k", 2).version == 1  # history forgotten, as documented


# ------------------------------------------------------------ apply_set (replica)
def test_apply_set_stores_the_primarys_version_and_absolute_expiry(cache, clock):
    result = cache.apply_set("k", {"a": 1}, version=7, created_at=clock.now - 5,
                             expires_at=clock.now + 30)
    assert result.applied and result.reason == APPLIED and result.version == 7

    clock.advance(10)  # time passing on the replica must count down the SAME deadline
    entry = cache.get("k")
    assert entry.version == 7 and entry.value == {"a": 1}
    assert entry.created_at == clock.now - 15
    assert entry.expires_at == clock.now + 20  # NOT restarted at 30 from receipt
    assert cache.ttl("k") == pytest.approx(20)


def test_apply_set_ignores_duplicate_and_older_versions(cache, clock):
    cache.apply_set("k", "v5", 5, clock.now, None)
    for version in (5, 4, 1):
        result = cache.apply_set("k", "old", version, clock.now, None)
        assert not result.applied and result.reason == STALE and result.version == 5
    assert cache.get("k").value == "v5"


def test_apply_set_accepts_newer_versions(cache, clock):
    cache.apply_set("k", "v1", 1, clock.now, None)
    assert cache.apply_set("k", "v2", 2, clock.now, clock.now + 10).applied
    entry = cache.get("k")
    assert (entry.value, entry.version) == ("v2", 2)
    assert entry.expires_at == clock.now + 10


def test_apply_set_with_an_already_passed_expiry_is_not_stored(cache, clock):
    result = cache.apply_set("k", "v", 1, clock.now - 10, clock.now - 1)
    assert not result.applied and result.reason == EXPIRED
    assert cache.size() == 0


def test_replicated_writes_are_not_counted_as_client_operations(cache, clock):
    cache.apply_set("k", "v", 1, clock.now, None)
    cache.apply_delete("k", 2)
    s = cache.stats
    assert (s.sets, s.gets, s.deletes, s.hits, s.misses) == (0, 0, 0, 0, 0)


def test_replicated_entries_obey_capacity_and_lru(cache, clock):  # capacity = 3
    for i in range(5):
        cache.apply_set(f"k{i}", i, 1, clock.now, None)
    assert cache.size() == 3
    assert cache.stats.evictions == 2
    assert not cache.exists("k0") and cache.exists("k4")


# ---------------------------------------------------------- apply_delete (replica)
def test_apply_delete_removes_the_key_and_blocks_older_sets(cache, clock):
    cache.apply_set("k", "v", 1, clock.now, None)
    assert cache.apply_delete("k", 2).applied
    assert not cache.exists("k")

    # a delayed SET v1 arrives after the delete: it must NOT bring the key back
    late = cache.apply_set("k", "zombie", 1, clock.now, None)
    assert not late.applied and late.reason == STALE
    assert not cache.exists("k")


def test_a_newer_set_after_a_replicated_delete_is_accepted(cache, clock):
    cache.apply_set("k", "v", 1, clock.now, None)
    cache.apply_delete("k", 2)
    assert cache.apply_set("k", "reborn", 3, clock.now, None).applied
    assert cache.get("k").value == "reborn"


def test_apply_delete_is_idempotent_and_ignores_stale_deletes(cache, clock):
    cache.apply_set("k", "v3", 3, clock.now, None)
    stale = cache.apply_delete("k", 2)  # an old delete must not remove newer data
    assert not stale.applied and stale.reason == STALE
    assert cache.get("k").value == "v3"

    assert cache.apply_delete("k", 4).applied
    again = cache.apply_delete("k", 4)  # duplicate delivery
    assert not again.applied and again.reason == STALE


def test_apply_delete_for_a_key_never_seen_still_leaves_a_tombstone(cache, clock):
    """The delete can overtake the SET it follows; the late SET must lose."""
    assert cache.apply_delete("k", 2).applied
    assert not cache.apply_set("k", "late", 1, clock.now, None).applied


# ------------------------------------------------------------- VersionManager
def test_tombstone_count_is_bounded(clock):
    vm = VersionManager(retention=100, max_entries=3, clock=clock)
    for i in range(10):
        vm.record(f"k{i}", i + 1)
    assert len(vm) == 3
    assert vm.floor("k0") == 0 and vm.floor("k9") == 10  # oldest dropped, newest kept


def test_purge_removes_only_expired_tombstones(clock):
    vm = VersionManager(retention=10, clock=clock)
    vm.record("old", 1)
    clock.advance(6)
    vm.record("new", 1)
    clock.advance(6)  # "old" is 12s old (expired), "new" is 6s old
    assert vm.purge() == 1
    assert vm.floor("new") == 1 and vm.floor("old") == 0


def test_record_never_lowers_a_floor(clock):
    vm = VersionManager(clock=clock)
    vm.record("k", 5)
    vm.record("k", 3)
    assert vm.floor("k") == 5


def test_engine_purge_also_clears_expired_tombstones(clock):
    cache = CacheEngine(capacity=5, clock=clock, tombstone_ttl=10)
    cache.set("k", 1)
    cache.delete("k")
    assert len(cache.versions) == 1
    clock.advance(11)
    cache.purge_expired()
    assert len(cache.versions) == 0
