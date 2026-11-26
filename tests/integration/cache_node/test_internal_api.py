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
