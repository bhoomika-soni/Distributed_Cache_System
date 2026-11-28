"""Real failures against a real cluster: kill node processes, watch the gateway cope.

One module-scoped 3-node cluster with replication factor 3 (every node holds every
key). Each test that kills a node uses `node_down`, which always restarts it and
waits until the gateway reports it healthy again, so tests stay independent.

Health checks run every 0.3s and 2 failures mark a node unhealthy, so detection
takes roughly a second.
"""
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from harness import ROOT, SECRET, load_module_from_file, running_cluster, wait_until  # noqa: E402

ch = load_module_from_file("gateway_consistent_hash_ft", ROOT / "gateway/app/services/consistent_hash.py")
HEADERS = {"X-Cluster-Secret": SECRET}
NODES = ("node-1", "node-2", "node-3")


def ring_order(key):
    """Primary first, then replicas - computed with the same ring the gateway uses."""
    ring = ch.HashRing(100)
    for n in NODES:
        ring.add_node(n)
    return ring.get_nodes(key, 3)


def key_with_primary(primary):
    for i in range(2000):
        key = f"ft-{uuid.uuid4().hex[:6]}-{i}"
        if ring_order(key)[0] == primary:
            return key
    raise AssertionError("no suitable key found")


def path(key):
    return f"/cache/{quote(key, safe='')}"


@pytest.fixture(scope="module")
def cluster():
    with running_cluster(node_count=3, replication_factor=3, request_timeout=1.0) as c:
        yield c


@pytest.fixture
def client(cluster):
    with httpx.Client(base_url=cluster.gateway_url, timeout=15) as c:
        yield c


def statuses(client):
    return {n["node_id"]: n["status"] for n in client.get("/cluster/nodes").json()}


def node_status(client, node_id):
    return statuses(client)[node_id]


def wait_all_healthy(client):
    wait_until(lambda: set(statuses(client).values()) == {"healthy"}, timeout=30, what="all nodes healthy")


@contextmanager
def node_down(cluster, client, node_id, wait_detected=True):
    """Crash a node; always bring it back (empty) and wait for full recovery afterwards."""
    cluster.stop_node(node_id)
    try:
        if wait_detected:
            wait_until(lambda: node_status(client, node_id) == "unhealthy", timeout=15, what=f"{node_id} detected")
        yield
    finally:
        cluster.restart_node(node_id)
        wait_all_healthy(client)


def node_get(cluster, node_id, key):
    r = httpx.get(f"{cluster.node_urls[node_id]}/internal/cache/get/{quote(key, safe='')}", headers=HEADERS)
    return r.json() if r.status_code == 200 else None


def replicated(cluster, key, nodes=NODES):
    return all(node_get(cluster, n, key) is not None for n in nodes)


# --------------------------------------------------------------------- baseline
def test_all_nodes_become_healthy_and_the_cluster_reports_it(client):
    wait_all_healthy(client)
    body = client.get("/cluster/health").json()
    assert body["status"] == "healthy" and body["healthy_nodes"] == 3 and body["total_nodes"] == 3


# -------------------------------------------------------------------- detection
def test_a_crashed_node_is_detected_and_the_cluster_reports_degraded(cluster, client):
    with node_down(cluster, client, "node-2"):
        body = client.get("/cluster/health").json()
        assert body["status"] == "degraded" and body["healthy_nodes"] == 2
        node = client.get("/cluster/nodes/node-2").json()
        assert node["status"] == "unhealthy" and node["consecutive_failures"] >= 2 and node["last_error"]
        assert client.get("/health").status_code == 200  # the gateway itself is fine
    assert node_status(client, "node-2") == "healthy"


# ------------------------------------------------------------- read fallback
def test_reads_fall_back_to_a_replica_when_the_primary_is_down(cluster, client):
    key = key_with_primary("node-1")
    client.post("/cache", json={"key": key, "value": {"v": 1}, "ttl": 300})
    wait_until(lambda: replicated(cluster, key), what="replication")

    with node_down(cluster, client, "node-1"):
        got = client.get(path(key))
        assert got.status_code == 200
        body = got.json()
        assert body["value"] == {"v": 1} and body["fallback"] is True and body["node_id"] != "node-1"
        assert client.get(path(key) + "/exists").json()["exists"] is True
        assert client.get(path(key) + "/ttl").json()["ttl_seconds"] > 0


def test_reads_fall_back_even_before_the_failure_is_detected(cluster, client):
    """Passive detection: the first request that hits the dead node moves on by itself."""
    key = key_with_primary("node-3")
    client.post("/cache", json={"key": key, "value": "x"})
    wait_until(lambda: replicated(cluster, key), what="replication")

    with node_down(cluster, client, "node-3", wait_detected=False):
        r = client.get(path(key))
        assert r.status_code == 200 and r.json()["value"] == "x" and r.json()["fallback"] is True


def test_a_miss_is_a_miss_not_a_reason_to_search_replicas(client):
    r = client.get(path(f"never-{uuid.uuid4().hex}"))
    assert r.status_code == 404 and "not found" in r.json()["detail"]


# ------------------------------------------------------------ write failover
def test_writes_fail_over_when_the_primary_is_down(cluster, client):
    key = key_with_primary("node-1")
    with node_down(cluster, client, "node-1"):
        r = client.post("/cache", json={"key": key, "value": "while-down"})
        assert r.status_code == 201
        body = r.json()
        assert body["fallback"] is True and body["node_id"] != "node-1"
        assert "node-1" not in body["replicas"]  # a dead node is never promised a copy

        survivor = body["node_id"]
        assert node_get(cluster, survivor, key)["value"] == "while-down"
        assert client.get(path(key)).json()["value"] == "while-down"


def test_deletes_fail_over_when_the_primary_is_down(cluster, client):
    key = key_with_primary("node-2")
    client.post("/cache", json={"key": key, "value": 1})
    wait_until(lambda: replicated(cluster, key), what="replication")

    with node_down(cluster, client, "node-2"):
        r = client.delete(path(key))
        assert r.status_code == 200 and r.json()["deleted"] is True and r.json()["fallback"] is True
        survivors = [n for n in NODES if n != "node-2"]
        wait_until(lambda: all(node_get(cluster, n, key) is None for n in survivors), what="delete replicated")
        assert client.get(path(key)).status_code == 404


# ------------------------------------------------------------------- recovery
def test_a_restarted_node_is_resynced_before_it_serves_again(cluster, client):
    old_key, new_key = key_with_primary("node-1"), key_with_primary("node-1")
    client.post("/cache", json={"key": old_key, "value": "before"})
    wait_until(lambda: replicated(cluster, old_key), what="replication")
    old_version = node_get(cluster, "node-2", old_key)["version"]

    cluster.stop_node("node-1")
    wait_until(lambda: node_status(client, "node-1") == "unhealthy", timeout=15, what="detected")
    client.post("/cache", json={"key": new_key, "value": "during"})  # written while node-1 was down

    cluster.restart_node("node-1")  # comes back EMPTY
    wait_all_healthy(client)

    # It was re-filled from its peers, with the same versions (so later writes continue the numbering).
    old = node_get(cluster, "node-1", old_key)
    new = node_get(cluster, "node-1", new_key)
    assert old and old["value"] == "before" and old["version"] == old_version
    assert new and new["value"] == "during"
    resync = client.get("/cluster/nodes/node-1").json()["last_resync"]
    assert resync["ok"] is True and resync["copied"] >= 2

    # The ring primary serves its own keys again - and a recovered primary continues the version count.
    got = client.get(path(old_key)).json()
    assert got["value"] == "before" and got["fallback"] is False and got["node_id"] == "node-1"
    assert client.post("/cache", json={"key": old_key, "value": "after"}).json()["version"] == old_version + 1


# ------------------------------------------------------------ multiple failures
def test_the_cluster_keeps_working_with_two_of_three_nodes_down(cluster, client):
    key = f"two-down-{uuid.uuid4().hex[:6]}"
    client.post("/cache", json={"key": key, "value": "alive"})
    wait_until(lambda: replicated(cluster, key), what="replication")

    down = ("node-1", "node-2")
    for n in down:
        cluster.stop_node(n)
    try:
        wait_until(lambda: all(node_status(client, n) == "unhealthy" for n in down), timeout=20, what="both detected")
        assert client.get("/cluster/health").json()["healthy_nodes"] == 1
        assert client.get(path(key)).json()["value"] == "alive"
        assert client.post("/cache", json={"key": key, "value": "still-writable"}).status_code in (200, 201)
    finally:
        for n in down:  # restart BOTH before waiting, otherwise the wait can never succeed
            cluster.restart_node(n)
        wait_all_healthy(client)


def test_with_every_node_down_the_gateway_says_503_but_stays_up_and_recovers(cluster, client):
    key = f"all-down-{uuid.uuid4().hex[:6]}"
    for n in NODES:
        cluster.stop_node(n)
    try:
        wait_until(lambda: set(statuses(client).values()) == {"unhealthy"}, timeout=20, what="all unhealthy")

        r = client.get(path(key))
        assert r.status_code == 503 and "node-1=unhealthy" in r.json()["detail"]
        assert client.post("/cache", json={"key": key, "value": 1}).status_code == 503

        health = client.get("/cluster/health")
        assert health.status_code == 503 and health.json()["status"] == "unavailable"
        assert client.get("/health").status_code == 200  # liveness of the gateway itself
    finally:
        for n in NODES:
            cluster.restart_node(n)
        wait_all_healthy(client)

    # Back in service (nodes restarted empty, so the key is simply a miss) and writable.
    assert client.get(path(key)).status_code == 404
    assert client.post("/cache", json={"key": key, "value": 1}).status_code == 201
