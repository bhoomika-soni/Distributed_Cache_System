"""Unit tests for the consistent hash ring.

Hashing is deterministic (MD5), so the key sets below always produce the same
results: these tests cannot flake.
"""
import hashlib
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from harness import ROOT, load_module_from_file  # noqa: E402

ch = load_module_from_file("gateway_consistent_hash", ROOT / "gateway/app/services/consistent_hash.py")
HashRing = ch.HashRing

KEYS = [f"key:{i}" for i in range(20000)]


def make_ring(node_ids=("node-1", "node-2", "node-3"), virtual_nodes=100):
    ring = HashRing(virtual_nodes)
    for node_id in node_ids:
        ring.add_node(node_id)
    return ring


def owners(ring, keys=KEYS):
    return {k: ring.get_node(k) for k in keys}


# ------------------------------------------------------------------ basics
def test_hash_is_the_deterministic_md5_not_pythons_randomized_hash():
    """Python's hash() differs between processes; MD5 must give the same answer everywhere."""
    expected = int.from_bytes(hashlib.md5(b"user:1").digest()[:8], "big")
    assert ch.hash_key("user:1") == expected


def test_same_key_always_maps_to_the_same_node():
    ring = make_ring()
    assert len({ring.get_node("user:101") for _ in range(50)}) == 1


def test_mapping_does_not_depend_on_the_order_nodes_were_added():
    a = make_ring(("node-1", "node-2", "node-3"))
    b = make_ring(("node-3", "node-1", "node-2"))
    assert owners(a, KEYS[:2000]) == owners(b, KEYS[:2000])


def test_empty_ring_raises():
    with pytest.raises(LookupError):
        HashRing(10).get_node("k")
    with pytest.raises(LookupError):
        HashRing(10).get_nodes("k", 1)


def test_single_node_owns_everything():
    ring = make_ring(("only",))
    assert set(owners(ring, KEYS[:500]).values()) == {"only"}


def test_ring_has_nodes_times_virtual_nodes_points():
    assert len(make_ring(virtual_nodes=50)) == 150


def test_invalid_arguments():
    with pytest.raises(ValueError):
        HashRing(0)
    ring = make_ring()
    with pytest.raises(ValueError):
        ring.add_node("node-1")  # duplicate
    with pytest.raises(KeyError):
        ring.remove_node("ghost")
    with pytest.raises(ValueError):
        ring.get_nodes("k", 0)


# ------------------------------------------------- distribution & virtual nodes
def spread(ring):
    counts = Counter(owners(ring).values())
    return counts, max(counts.values()) / min(counts.values())


def test_virtual_nodes_spread_keys_evenly():
    counts, ratio = spread(make_ring(virtual_nodes=100))
    assert set(counts) == {"node-1", "node-2", "node-3"}
    for node_id, n in counts.items():
        assert 0.25 < n / len(KEYS) < 0.42, f"{node_id} got {n / len(KEYS):.1%}"
    assert ratio < 1.5


def test_a_single_point_per_node_is_badly_unbalanced():
    """This is WHY virtual nodes exist."""
    _, ratio_one = spread(make_ring(virtual_nodes=1))
    _, ratio_many = spread(make_ring(virtual_nodes=100))
    assert ratio_one > 3
    assert ratio_many < ratio_one / 2


# --------------------------------------------------- minimal key movement (the point)
def test_adding_a_node_moves_few_keys_and_only_to_the_new_node():
    ring = make_ring()
    before = owners(ring)
    ring.add_node("node-4")
    after = owners(ring)

    moved = [k for k in KEYS if before[k] != after[k]]
    assert all(after[k] == "node-4" for k in moved)  # nothing shuffles between old nodes
    assert 0.10 < len(moved) / len(KEYS) < 0.35  # ideal is 1/4

    # For contrast: the naive scheme hash(key) % N reshuffles most keys.
    def naive(key, n):
        return ch.hash_key(key) % n

    naive_moved = sum(naive(k, 3) != naive(k, 4) for k in KEYS) / len(KEYS)
    assert naive_moved > 0.6
    assert len(moved) / len(KEYS) < naive_moved / 2


def test_removing_a_node_only_moves_that_nodes_keys_and_spreads_them():
    ring = make_ring()
    before = owners(ring)
    ring.remove_node("node-2")
    after = owners(ring)

    moved = [k for k in KEYS if before[k] != after[k]]
    assert moved and all(before[k] == "node-2" for k in moved)  # only node-2's keys moved
    assert all(after[k] != "node-2" for k in KEYS)
    # Virtual nodes hand the orphaned keys to BOTH remaining nodes, not one neighbour.
    received = Counter(after[k] for k in moved)
    assert set(received) == {"node-1", "node-3"}
    assert min(received.values()) / len(moved) > 0.3


def test_add_then_remove_restores_the_original_mapping():
    ring = make_ring()
    before = owners(ring)
    ring.add_node("node-4")
    ring.remove_node("node-4")
    assert owners(ring) == before


# ------------------------------------------------------- get_nodes: replicas / failover
def test_get_nodes_returns_distinct_nodes_starting_with_the_primary():
    ring = make_ring()
    for key in KEYS[:300]:
        nodes = ring.get_nodes(key, 3)
        assert nodes[0] == ring.get_node(key)
        assert len(nodes) == len(set(nodes)) == 3


def test_get_nodes_never_returns_more_nodes_than_exist():
    ring = make_ring(("node-1", "node-2"))
    assert len(ring.get_nodes("k", 5)) == 2


def test_get_nodes_skips_unavailable_nodes():
    ring = make_ring()
    for key in KEYS[:300]:
        nodes = ring.get_nodes(key, 3, available={"node-1", "node-3"})
        assert "node-2" not in nodes and len(nodes) == 2


def test_next_available_node_equals_the_owner_on_a_ring_without_the_failed_node():
    """Skipping a down node on the ring gives exactly the answer you'd get if the
    node were removed - but the ring itself is untouched, so when the node comes
    back, keys are found where they were."""
    full = make_ring()
    without = make_ring(("node-1", "node-3"))
    for key in KEYS[:2000]:
        assert full.get_nodes(key, 1, available={"node-1", "node-3"})[0] == without.get_node(key)


def test_no_available_nodes_returns_an_empty_list():
    assert make_ring().get_nodes("k", 2, available=set()) == []
