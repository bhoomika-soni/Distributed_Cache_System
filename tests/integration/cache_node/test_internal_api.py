import time

import pytest


def put(client, key, value="v", **extra):
    return client.post("/internal/cache/set", json={"key": key, "value": value, **extra})


# ----------------------------------------------------------------- security
ENDPOINTS = [
    ("GET", "/internal/health", None),
    ("POST", "/internal/cache/set", {"key": "k", "value": 1}),
    ("GET", "/internal/cache/get/k", None),
    ("DELETE", "/internal/cache/delete/k", None),
    ("GET", "/internal/cache/exists/k", None),
    ("GET", "/internal/cache/ttl/k", None),
    ("GET", "/internal/replication/stats", None),
    ("POST", "/internal/replication/apply",
     {"op": "delete", "key": "k", "version": 1, "origin": "n", "sent_at": 1.0}),
]


@pytest.mark.parametrize("method, path, body", ENDPOINTS)
def test_every_endpoint_rejects_missing_secret(anon_client, method, path, body):
    assert anon_client.request(method, path, json=body).status_code == 401


@pytest.mark.parametrize("method, path, body", ENDPOINTS)
def test_every_endpoint_rejects_wrong_secret(anon_client, method, path, body):
    r = anon_client.request(method, path, json=body, headers={"X-Cluster-Secret": "nope"})
    assert r.status_code == 401


def test_rejected_write_does_not_change_state(client, anon_client):
    anon_client.post("/internal/cache/set", json={"key": "sneaky", "value": 1})
    assert client.get("/internal/cache/get/sneaky").status_code == 404


def test_health_reports_real_key_count(client):
    assert client.get("/internal/health").json()["keys"] == 0
    put(client, "a")
    body = client.get("/internal/health").json()
    assert body["status"] == "up" and body["node_id"] == "test-node" and body["keys"] == 1


# ------------------------------------------------------------ set / get
def test_set_then_get_roundtrip_with_nested_json(client):
    value = {"name": "John", "tags": ["a", "b"], "meta": {"age": 30, "ok": True}}
    r = put(client, "user:101", value, ttl=60)
    assert r.status_code == 200
    assert r.json()["created"] is True and r.json()["version"] == 1

    got = client.get("/internal/cache/get/user:101").json()
    assert got["value"] == value
    assert got["version"] == 1
    assert 0 < got["ttl_seconds"] <= 60
    assert got["expires_at"] is not None


def test_overwrite_increments_version(client):
    put(client, "k", "one")
    second = put(client, "k", "two").json()
    assert second["created"] is False and second["version"] == 2
    assert client.get("/internal/cache/get/k").json()["value"] == "two"


def test_stored_null_is_a_hit_not_a_miss(client):
    put(client, "k", None)
    r = client.get("/internal/cache/get/k")
    assert r.status_code == 200 and r.json()["value"] is None


def test_get_missing_key_is_404(client):
    r = client.get("/internal/cache/get/nope")
    assert r.status_code == 404
    assert "nope" in r.json()["detail"]


def test_keys_containing_slashes_work_on_every_endpoint(client):
    put(client, "a/b/c", 1)
    assert client.get("/internal/cache/get/a/b/c").json()["value"] == 1
    assert client.get("/internal/cache/exists/a/b/c").json()["exists"] is True
    assert client.get("/internal/cache/ttl/a/b/c").json()["state"] == "no_expiry"
    assert client.delete("/internal/cache/delete/a/b/c").json()["deleted"] is True


# ------------------------------------------------------------- delete / exists
def test_delete_is_idempotent(client):
    put(client, "k")
    first = client.delete("/internal/cache/delete/k")
    second = client.delete("/internal/cache/delete/k")
    assert (first.status_code, first.json()["deleted"]) == (200, True)
    assert (second.status_code, second.json()["deleted"]) == (200, False)
    assert client.get("/internal/cache/get/k").status_code == 404


def test_exists(client):
    assert client.get("/internal/cache/exists/k").json() == {"key": "k", "exists": False}
    put(client, "k")
    assert client.get("/internal/cache/exists/k").json() == {"key": "k", "exists": True}


# ------------------------------------------------------------------------ TTL
def test_ttl_states(client):
    assert client.get("/internal/cache/ttl/k").json() == {
        "key": "k", "state": "missing", "ttl_seconds": None,
    }
    put(client, "forever")
    assert client.get("/internal/cache/ttl/forever").json()["state"] == "no_expiry"

    put(client, "temp", ttl=100)
    body = client.get("/internal/cache/ttl/temp").json()
    assert body["state"] == "expiring" and 0 < body["ttl_seconds"] <= 100


def test_expired_key_is_a_miss_over_http(client):
    put(client, "k", ttl=0.1)
    time.sleep(0.2)
    assert client.get("/internal/cache/get/k").status_code == 404
    assert client.get("/internal/cache/ttl/k").json()["state"] == "missing"


def test_background_cleaner_runs_inside_the_node(client):
    """Cleanup interval is 0.05s in these tests; no GET is sent for the key."""
    put(client, "k", ttl=0.05)
    time.sleep(0.4)
    engine = client.app.state.engine
    assert engine.size() == 0
    assert engine.stats.gets == 0  # removed by the cleaner, not by a lazy read


# ----------------------------------------------------------------- LRU / capacity
def test_lru_eviction_through_the_api(client):  # capacity is 3
    for k in ("a", "b", "c"):
        put(client, k)
    client.get("/internal/cache/get/a")  # "b" becomes least recently used
    put(client, "d")

    assert client.get("/internal/cache/exists/b").json()["exists"] is False
    for k in ("a", "c", "d"):
        assert client.get(f"/internal/cache/exists/{k}").json()["exists"] is True
    assert client.app.state.engine.stats.evictions == 1


# ----------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "payload",
    [
        {"key": "", "value": 1},  # empty key
        {"key": "k" * 513, "value": 1},  # key too long
        {"key": "k"},  # value missing
        {"value": 1},  # key missing
        {"key": "k", "value": 1, "ttl": 0},  # ttl must be > 0
        {"key": "k", "value": 1, "ttl": -5},
        {"key": "k", "value": 1, "ttl": "soon"},
        {"key": "k", "value": 1, "TTL": 60},  # misspelled field is rejected, not ignored
    ],
)
def test_invalid_set_payloads_are_422_and_store_nothing(client, payload):
    assert client.post("/internal/cache/set", json=payload).status_code == 422
    assert client.app.state.engine.size() == 0


def test_malformed_json_is_422(client):
    r = client.post(
        "/internal/cache/set",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422


# ------------------------------------------------------- replication (Phase 6)
UNREACHABLE = {"node_id": "ghost", "url": "http://127.0.0.1:1"}  # nothing listens on port 1


def stats(client):
    return client.get("/internal/replication/stats").json()


def apply(client, **payload):
    return client.post("/internal/replication/apply", json=payload)


def wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_set_without_replicas_schedules_no_replication(client):
    body = put(client, "k").json()
    assert body["replicas_scheduled"] == []
    assert stats(client)["sent"]["scheduled"] == 0


def test_set_reports_the_replicas_it_scheduled_without_waiting_for_them(client):
    started = time.time()
    r = put(client, "k", replicas=[UNREACHABLE])
    assert r.status_code == 200 and r.json()["replicas_scheduled"] == ["ghost"]
    assert time.time() - started < 0.25  # asynchronous: the client does not wait for the replica
    assert stats(client)["sent"]["scheduled"] == 1


def test_unreachable_replica_is_recorded_as_a_failure_not_dropped(client):
    put(client, "k", "v", replicas=[UNREACHABLE])
    assert wait_for(lambda: stats(client)["sent"]["failed"] == 1), "failure was never recorded"

    s = stats(client)
    failure = s["recent_failures"][0]
    assert (failure["op"], failure["key"], failure["version"], failure["target"]) == (
        "set", "k", 1, "ghost")
    assert failure["attempts"] == 2  # REPLICATION_RETRIES=1 in these tests: 1 try + 1 retry
    assert s["sent"]["succeeded"] == 0 and s["sent"]["retries"] == 1
    assert s["sent"]["in_flight"] == 0
    # the primary's own copy is unaffected by the failed replication
    assert client.get("/internal/cache/get/k").json()["value"] == "v"


def test_delete_with_replicas_replicates_only_real_deletes(client):
    put(client, "k")
    r = client.request("DELETE", "/internal/cache/delete/k", json={"replicas": [UNREACHABLE]})
    assert r.json() == {"key": "k", "deleted": True, "replicas_scheduled": ["ghost"]}

    missing = client.request("DELETE", "/internal/cache/delete/nope", json={"replicas": [UNREACHABLE]})
    assert missing.json()["deleted"] is False and missing.json()["replicas_scheduled"] == []


def test_invalid_replica_targets_are_422(client):
    for bad in ([{"node_id": "", "url": "http://h"}], [{"node_id": "n", "url": "ftp://h"}], [{"node_id": "n"}]):
        assert put(client, "k", replicas=bad).status_code == 422


def test_apply_stores_the_primarys_version_and_absolute_expiry(client):
    expires_at = time.time() + 100
    r = apply(client, op="set", key="k", value={"a": 1}, version=7, created_at=time.time() - 5,
              expires_at=expires_at, origin="node-1", sent_at=time.time())
    assert r.status_code == 200
    assert r.json() == {"applied": True, "reason": "applied", "version": 7}

    got = client.get("/internal/cache/get/k").json()
    assert got["version"] == 7 and got["value"] == {"a": 1}
    assert got["expires_at"] == expires_at  # not restarted from receipt time


def test_apply_ignores_stale_versions_and_blocks_resurrection(client):
    base = dict(key="k", origin="node-1", sent_at=time.time())
    apply(client, op="set", value="v5", version=5, created_at=1.0, **base)

    stale = apply(client, op="set", value="old", version=3, created_at=1.0, **base).json()
    assert stale == {"applied": False, "reason": "stale", "version": 5}

    assert apply(client, op="delete", version=6, **base).json()["applied"] is True
    assert client.get("/internal/cache/get/k").status_code == 404
    zombie = apply(client, op="set", value="zombie", version=4, created_at=1.0, **base).json()
    assert zombie["reason"] == "stale"
    assert client.get("/internal/cache/get/k").status_code == 404


def test_primary_continues_numbering_after_a_delete(client):
    assert put(client, "k").json()["version"] == 1
    client.delete("/internal/cache/delete/k")
    assert put(client, "k").json()["version"] == 3  # 1 = set, 2 = delete, 3 = new set


def test_apply_counts_what_it_received_and_never_forwards(client):
    base = dict(key="k", origin="node-1", sent_at=time.time() - 0.2)
    apply(client, op="set", value=1, version=2, created_at=1.0, **base)  # applied
    apply(client, op="set", value=1, version=2, created_at=1.0, **base)  # duplicate -> stale
    apply(client, op="set", value=1, version=9, created_at=1.0, expires_at=time.time() - 1, **base)  # expired

    s = stats(client)
    assert s["received"]["total"] == 3
    assert (s["received"]["applied"], s["received"]["ignored_stale"], s["received"]["ignored_expired"]) == (1, 1, 1)
    assert s["received"]["last_lag_seconds"] >= 0.2
    assert s["sent"]["scheduled"] == 0  # a replica never replicates onward


@pytest.mark.parametrize(
    "payload",
    [
        {"op": "set", "key": "k", "version": 1, "origin": "n", "sent_at": 1.0, "created_at": 1.0},
        {"op": "set", "key": "k", "version": 1, "origin": "n", "sent_at": 1.0, "value": 1},
        {"op": "delete", "key": "k", "version": 1, "origin": "n", "sent_at": 1.0, "value": 1},
        {"op": "merge", "key": "k", "version": 1, "origin": "n", "sent_at": 1.0},
        {"op": "delete", "key": "k", "version": 0, "origin": "n", "sent_at": 1.0},
        {"op": "delete", "key": "", "version": 1, "origin": "n", "sent_at": 1.0},
    ],
)
def test_malformed_replication_messages_are_422(client, payload):
    assert client.post("/internal/replication/apply", json=payload).status_code == 422


# ------------------------------------------------------------------ snapshot
def test_snapshot_lists_live_entries_with_versions_and_absolute_expiry(client):
    client.post("/internal/cache/set", json={"key": "a", "value": 1})
    client.post("/internal/cache/set", json={"key": "a", "value": 2})  # version 2
    client.post("/internal/cache/set", json={"key": "b", "value": {"x": 1}, "ttl": 100})
    client.post("/internal/cache/set", json={"key": "gone", "value": 1})
    client.delete("/internal/cache/delete/gone")

    body = client.get("/internal/cache/snapshot").json()
    entries = {e["key"]: e for e in body["entries"]}
    assert body["count"] == 2 and set(entries) == {"a", "b"}  # deleted keys are not included
    assert entries["a"]["version"] == 2 and entries["a"]["value"] == 2
    assert entries["a"]["expires_at"] is None and entries["b"]["expires_at"] is not None


def test_snapshot_requires_the_cluster_secret(client):
    r = client.get("/internal/cache/snapshot", headers={"X-Cluster-Secret": "wrong"})
    assert r.status_code == 401
