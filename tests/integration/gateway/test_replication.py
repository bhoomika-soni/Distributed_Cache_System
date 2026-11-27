"""End-to-end replication: three real nodes, replication factor 3 (and 2)."""
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from harness import ROOT, SECRET, load_module_from_file, wait_until  # noqa: E402

ch = load_module_from_file("gateway_consistent_hash_repl", ROOT / "gateway/app/services/consistent_hash.py")

HEADERS = {"X-Cluster-Secret": SECRET}
NODE_IDS = ["node-1", "node-2", "node-3"]


def ring():
    r = ch.HashRing(100)  # the gateway's default VIRTUAL_NODES
    for node_id in NODE_IDS:
        r.add_node(node_id)
    return r


def unique(prefix="rep"):
    return f"{prefix}:{uuid.uuid4().hex[:8]}"


def path(key):
    return f"/cache/{quote(key, safe='')}"


# ------------------------------------------------------- talking to nodes directly
def node_get(url, key):
    r = httpx.get(f"{url}/internal/cache/get/{quote(key, safe='')}", headers=HEADERS)
    return r.json() if r.status_code == 200 else None


def holders(cluster, key):
    """{node_id: stored entry} for every node that currently holds `key`."""
    found = {nid: node_get(url, key) for nid, url in cluster.node_urls.items()}
    return {nid: entry for nid, entry in found.items() if entry is not None}


def stats(url):
    return httpx.get(f"{url}/internal/replication/stats", headers=HEADERS).json()


def apply(url, **payload):
    return httpx.post(f"{url}/internal/replication/apply", json=payload, headers=HEADERS)


def totals(cluster):
    snaps = [stats(url) for url in cluster.node_urls.values()]
    return {
        "scheduled": sum(s["sent"]["scheduled"] for s in snaps),
        "succeeded": sum(s["sent"]["succeeded"] for s in snaps),
        "failed": sum(s["sent"]["failed"] for s in snaps),
        "received": sum(s["received"]["total"] for s in snaps),
        "applied": sum(s["received"]["applied"] for s in snaps),
    }


def settle(cluster):
    """Wait until no replication is in flight anywhere (so counters stop moving)."""
    def quiet():
        snaps = [stats(url) for url in cluster.node_urls.values()]
        return all(s["sent"]["in_flight"] == 0 for s in snaps)
    wait_until(quiet, what="all in-flight replications to finish")


# ------------------------------------------------------------------ replication
def test_set_is_replicated_to_every_replica_with_the_same_version(replicated_client, replicated_cluster):
    key = unique()
    r = replicated_client.post("/cache", json={"key": key, "value": {"n": 1}})
    body = r.json()

    # The gateway answers once the PRIMARY has the write and says replication is async.
    expected = ring().get_nodes(key, 3)
    assert r.status_code == 201 and body["replication"] == "async"
    assert body["node_id"] == expected[0] and body["replicas"] == expected[1:]

    wait_until(lambda: len(holders(replicated_cluster, key)) == 3, what="all 3 nodes to hold the key")
    copies = holders(replicated_cluster, key)
    assert all(c["value"] == {"n": 1} and c["version"] == 1 for c in copies.values())


def test_replicas_get_the_primarys_absolute_expiry_not_a_fresh_ttl(replicated_client, replicated_cluster):
    key = unique()
    replicated_client.post("/cache", json={"key": key, "value": 1, "ttl": 30})
    wait_until(lambda: len(holders(replicated_cluster, key)) == 3, what="replication")

    copies = holders(replicated_cluster, key)
    # If each replica restarted "30 seconds" on arrival, the expiry times would differ
    # by the replication delay. They must be exactly equal.
    assert len({c["expires_at"] for c in copies.values()}) == 1
    assert all(0 < c["ttl_seconds"] <= 30 for c in copies.values())


def test_overwrite_replicates_a_higher_version(replicated_client, replicated_cluster):
    key = unique()
    replicated_client.post("/cache", json={"key": key, "value": "first"})
    replicated_client.post("/cache", json={"key": key, "value": "second"})

    def converged():
        copies = holders(replicated_cluster, key)
        return len(copies) == 3 and all(
            c["version"] == 2 and c["value"] == "second" for c in copies.values()
        )

    wait_until(converged, what="all nodes to reach version 2")


def test_delete_is_replicated(replicated_client, replicated_cluster):
    key = unique()
    replicated_client.post("/cache", json={"key": key, "value": 1})
    wait_until(lambda: len(holders(replicated_cluster, key)) == 3, what="replication")

    r = replicated_client.delete(path(key))
    assert r.json()["deleted"] is True and r.json()["replication"] == "async"
    wait_until(lambda: holders(replicated_cluster, key) == {}, what="the delete to reach all nodes")


def test_set_after_delete_converges_on_every_node(replicated_client, replicated_cluster):
    """SET v1 -> DELETE (v2) -> SET (v3). Replicas must accept v3, not reject it as old."""
    key = unique()
    replicated_client.post("/cache", json={"key": key, "value": "first"})
    wait_until(lambda: len(holders(replicated_cluster, key)) == 3, what="first set")
    replicated_client.delete(path(key))
    wait_until(lambda: holders(replicated_cluster, key) == {}, what="delete")
    replicated_client.post("/cache", json={"key": key, "value": "second"})

    def converged():
        copies = holders(replicated_cluster, key)
        return len(copies) == 3 and all(
            c["value"] == "second" and c["version"] == 3 for c in copies.values()
        )

    wait_until(converged, what="all nodes to hold the re-created key at version 3")


def test_delete_of_a_missing_key_replicates_nothing(replicated_client, replicated_cluster):
    settle(replicated_cluster)
    before = totals(replicated_cluster)
    r = replicated_client.delete(path(unique("never-set")))
    assert r.json()["deleted"] is False
    time.sleep(0.3)
    assert totals(replicated_cluster)["scheduled"] == before["scheduled"]


def test_replication_factor_2_stores_the_key_on_exactly_two_nodes(rf2_client, replicated_cluster):
    for _ in range(5):
        key = unique()
        body = rf2_client.post("/cache", json={"key": key, "value": 1}).json()
        owners = ring().get_nodes(key, 2)
        assert [body["node_id"], *body["replicas"]] == owners

        wait_until(lambda: set(holders(replicated_cluster, key)) == set(owners),
                   what=f"{key} on exactly {owners}")
        time.sleep(0.2)  # a third copy, if one were wrongly sent, would show up by now
        assert set(holders(replicated_cluster, key)) == set(owners)


# ---------------------------------------------------------------------- tracking
def test_replication_is_tracked_with_real_counters(replicated_client, replicated_cluster):
    settle(replicated_cluster)
    before = totals(replicated_cluster)

    for _ in range(10):
        replicated_client.post("/cache", json={"key": unique(), "value": 1})
    # 10 writes x 2 replicas = 20 deliveries
    wait_until(lambda: totals(replicated_cluster)["succeeded"] - before["succeeded"] == 20,
               what="20 successful deliveries")

    after = totals(replicated_cluster)
    assert after["scheduled"] - before["scheduled"] == 20
    assert after["failed"] - before["failed"] == 0
    assert after["received"] - before["received"] == 20
    assert after["applied"] - before["applied"] == 20


# ------------------------------------------------------- loop prevention & apply
def test_a_replicated_write_is_never_forwarded_again(replicated_cluster):
    """Send a write straight to node-2's apply endpoint, as if from a primary.
    node-2 must store it and stop there: no further replication to node-1 or node-3."""
    settle(replicated_cluster)
    urls = replicated_cluster.node_urls
    before = {nid: stats(url) for nid, url in urls.items()}

    key = unique()
    r = apply(urls["node-2"], op="set", key=key, value="x", version=3, created_at=time.time(),
              expires_at=None, origin="node-1", sent_at=time.time())
    assert r.status_code == 200
    assert r.json() == {"applied": True, "reason": "applied", "version": 3}

    time.sleep(0.5)  # time enough for a (wrong) forward to arrive
    assert node_get(urls["node-2"], key)["version"] == 3
    assert node_get(urls["node-1"], key) is None and node_get(urls["node-3"], key) is None
    after = {nid: stats(url) for nid, url in urls.items()}
    assert after["node-2"]["sent"]["scheduled"] == before["node-2"]["sent"]["scheduled"]
    for nid in ("node-1", "node-3"):
        assert after[nid]["received"]["total"] == before[nid]["received"]["total"]


def test_apply_endpoint_enforces_versions_over_http(replicated_cluster):
    url = replicated_cluster.node_urls["node-3"]
    key = unique()
    base = dict(key=key, origin="node-1", created_at=time.time(), expires_at=None)

    def set_v(version, value="v"):
        return apply(url, op="set", value=value, version=version, sent_at=time.time(), **base).json()

    def delete_v(version):
        return apply(url, op="delete", key=key, version=version, origin="node-1",
                     sent_at=time.time()).json()

    assert set_v(5, "five")["applied"] is True
    assert set_v(3, "old") == {"applied": False, "reason": "stale", "version": 5}
    assert set_v(5, "dup") == {"applied": False, "reason": "stale", "version": 5}  # duplicate delivery
    assert node_get(url, key)["value"] == "five"

    assert delete_v(6)["applied"] is True and node_get(url, key) is None
    assert set_v(4, "zombie")["reason"] == "stale"  # a late old SET must not resurrect the key
    assert node_get(url, key) is None
    assert set_v(7, "reborn")["applied"] is True and node_get(url, key)["value"] == "reborn"


def test_apply_ignores_an_entry_that_has_already_expired(replicated_cluster):
    url = replicated_cluster.node_urls["node-3"]
    key = unique()
    r = apply(url, op="set", key=key, value=1, version=1, origin="node-1",
              created_at=time.time() - 10, expires_at=time.time() - 1, sent_at=time.time())
    assert r.json()["reason"] == "expired"
    assert node_get(url, key) is None


@pytest.mark.parametrize(
    "payload",
    [
        {"op": "set", "key": "k", "version": 1, "origin": "n", "sent_at": 1.0, "created_at": 1.0},  # no value
        {"op": "set", "key": "k", "version": 1, "origin": "n", "sent_at": 1.0, "value": 1},  # no created_at
        {"op": "delete", "key": "k", "version": 1, "origin": "n", "sent_at": 1.0, "value": 1},
        {"op": "merge", "key": "k", "version": 1, "origin": "n", "sent_at": 1.0},
        {"op": "delete", "key": "k", "version": 0, "origin": "n", "sent_at": 1.0},
    ],
)
def test_malformed_replication_messages_are_rejected(replicated_cluster, payload):
    assert apply(replicated_cluster.node_urls["node-1"], **payload).status_code == 422


def test_replication_endpoints_require_the_cluster_secret(replicated_cluster):
    url = replicated_cluster.node_urls["node-1"]
    body = {"op": "delete", "key": "k", "version": 1, "origin": "n", "sent_at": 1.0}
    assert httpx.post(f"{url}/internal/replication/apply", json=body).status_code == 401
    assert httpx.get(f"{url}/internal/replication/stats").status_code == 401
