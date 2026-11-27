"""End-to-end: three real nodes behind a gateway that routes with the hash ring."""
import sys
from pathlib import Path
from urllib.parse import quote

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from harness import ROOT, SECRET, load_module_from_file  # noqa: E402

ch = load_module_from_file("gateway_consistent_hash_e2e", ROOT / "gateway/app/services/consistent_hash.py")

# Fixed keys + fixed node ids + MD5 = the same placement every run (no flakiness).
KEYS = [f"route-test-{i}" for i in range(120)]
NODE_IDS = ["node-1", "node-2", "node-3"]


def expected_ring():
    ring = ch.HashRing(100)  # the gateway's default VIRTUAL_NODES
    for node_id in NODE_IDS:
        ring.add_node(node_id)
    return ring


def test_three_nodes_are_registered(three_node_client):
    assert three_node_client.get("/cluster").json()["node_ids"] == NODE_IDS


def test_keys_are_spread_across_all_three_nodes(three_node_client):
    placed = {}
    for key in KEYS:
        r = three_node_client.post("/cache", json={"key": key, "value": key.upper()})
        assert r.status_code == 201
        placed[key] = r.json()["node_id"]

    counts = {n: list(placed.values()).count(n) for n in NODE_IDS}
    assert all(c >= 15 for c in counts.values()), counts  # fair share is 40 each


def test_gateway_places_keys_exactly_where_the_ring_says(three_node_client):
    """Ties the e2e system to the unit-tested algorithm."""
    ring = expected_ring()
    for key in KEYS[:40]:
        r = three_node_client.post("/cache", json={"key": key, "value": 1})
        assert r.json()["node_id"] == ring.get_node(key)


def test_reads_go_to_the_same_node_that_stored_the_key(three_node_client):
    ring = expected_ring()
    for key in KEYS[:40]:
        three_node_client.post("/cache", json={"key": key, "value": {"k": key}})
        got = three_node_client.get(f"/cache/{quote(key, safe='')}").json()
        assert got["value"] == {"k": key}
        assert got["node_id"] == ring.get_node(key)


def test_each_key_is_stored_on_exactly_one_node(three_node_client, three_node_cluster):
    """No replication yet (Phase 6): the owner has the key and the other nodes do not."""
    ring = expected_ring()
    for key in KEYS[:20]:
        three_node_client.post("/cache", json={"key": key, "value": "x"})
        holders = []
        for node_id, url in three_node_cluster.node_urls.items():
            r = httpx.get(
                f"{url}/internal/cache/exists/{quote(key, safe='')}",
                headers={"X-Cluster-Secret": SECRET},
            )
            if r.json()["exists"]:
                holders.append(node_id)
        assert holders == [ring.get_node(key)]
