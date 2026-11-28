"""A replica is down: failed replication must be tracked and visible, never silent.

Own module (and cluster) because the tests STOP a node, which would break any
other test sharing it. Full failure handling (health checks, replica fallback
for reads, recovery) is Phase 7; this only covers replication bookkeeping.
"""
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from harness import ROOT, SECRET, load_module_from_file, running_cluster, wait_until  # noqa: E402

ch = load_module_from_file("gateway_consistent_hash_fail", ROOT / "gateway/app/services/consistent_hash.py")
HEADERS = {"X-Cluster-Secret": SECRET}
DEAD = "node-3"


@pytest.fixture(scope="module")
def cluster():
    # health tracking off: this module tests replication bookkeeping, not failover
    with running_cluster(node_count=3, replication_factor=3, health_check_interval=3600) as c:
        c.node_services[DEAD].stop()  # simulate a crash
        yield c


@pytest.fixture
def client(cluster):
    with httpx.Client(base_url=cluster.gateway_url, timeout=15) as c:
        yield c


def primary_and_live_replica(key):
    ring = ch.HashRing(100)
    for node_id in ("node-1", "node-2", "node-3"):
        ring.add_node(node_id)
    order = ring.get_nodes(key, 3)
    live = [n for n in order if n != DEAD]
    return live[0], live[1]  # (primary, the replica that is still alive)


def key_with_live_primary():
    """A key whose PRIMARY is alive (a dead primary is a Phase 7 topic)."""
    ring = ch.HashRing(100)
    for node_id in ("node-1", "node-2", "node-3"):
        ring.add_node(node_id)
    for i in range(1000):
        key = f"fail-key-{uuid.uuid4().hex[:6]}-{i}"
        if ring.get_nodes(key, 1)[0] != DEAD:
            return key
    raise AssertionError("no suitable key found")


def stats(cluster, node_id):
    return httpx.get(f"{cluster.node_urls[node_id]}/internal/replication/stats", headers=HEADERS).json()


def node_get(cluster, node_id, key):
    r = httpx.get(f"{cluster.node_urls[node_id]}/internal/cache/get/{quote(key, safe='')}", headers=HEADERS)
    return r.json() if r.status_code == 200 else None


def test_write_succeeds_and_the_failed_replication_is_recorded(client, cluster):
    key = key_with_live_primary()
    primary, live_replica = primary_and_live_replica(key)

    r = client.post("/cache", json={"key": key, "value": "important"})
    assert r.status_code == 201  # the primary has it, so the client is not blocked or failed
    assert r.json()["node_id"] == primary and DEAD in r.json()["replicas"]

    # The healthy replica still receives the write.
    wait_until(lambda: node_get(cluster, live_replica, key) is not None, what="live replica to get the key")

    # The dead replica's delivery fails after bounded retries and is RECORDED, not dropped.
    wait_until(lambda: any(f["key"] == key for f in stats(cluster, primary)["recent_failures"]),
               timeout=30, what="the failed replication to be recorded")
    s = stats(cluster, primary)
    failure = next(f for f in s["recent_failures"] if f["key"] == key)
    assert failure["target"] == DEAD and failure["op"] == "set" and failure["version"] == 1
    assert failure["attempts"] == 3  # 1 try + 2 retries, then it stopped
    assert s["sent"]["failed"] >= 1 and s["sent"]["retries"] >= 2


def test_failed_delete_replication_is_recorded_too(client, cluster):
    key = key_with_live_primary()
    primary, live_replica = primary_and_live_replica(key)
    client.post("/cache", json={"key": key, "value": 1})
    wait_until(lambda: node_get(cluster, live_replica, key) is not None, what="live replica to get the key")

    assert client.delete(f"/cache/{quote(key, safe='')}").json()["deleted"] is True
    wait_until(lambda: node_get(cluster, live_replica, key) is None, what="live replica to apply the delete")
    wait_until(
        lambda: any(f["key"] == key and f["op"] == "delete" for f in stats(cluster, primary)["recent_failures"]),
        timeout=30, what="the failed delete replication to be recorded",
    )


def test_the_gateway_stays_healthy_and_keeps_serving_reads_from_live_primaries(client):
    key = key_with_live_primary()
    client.post("/cache", json={"key": key, "value": "still here"})
    assert client.get("/health").status_code == 200
    assert client.get(f"/cache/{quote(key, safe='')}").json()["value"] == "still here"
