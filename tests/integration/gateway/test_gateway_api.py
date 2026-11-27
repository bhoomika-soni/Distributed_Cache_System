import time
import uuid
from urllib.parse import quote

import httpx
import pytest


def unique(prefix="k"):
    return f"{prefix}:{uuid.uuid4().hex[:8]}"


def path(key):
    return f"/cache/{quote(key, safe='')}"


# ----------------------------------------------------------- gateway & cluster
def test_gateway_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["configured_nodes"] == 2


def test_cluster_endpoints_list_the_registered_nodes(client, cluster):
    assert client.get("/cluster").json() == {"node_count": 2, "node_ids": ["node-1", "node-2"]}

    nodes = client.get("/cluster/nodes").json()
    assert [n["node_id"] for n in nodes] == ["node-1", "node-2"]
    assert nodes[0]["url"] == cluster.node_urls["node-1"]
    assert all(n["status"] == "unknown" for n in nodes)  # honest: no health checks yet

    assert client.get("/cluster/nodes/node-2").json()["node_id"] == "node-2"
    assert client.get("/cluster/nodes/node-9").status_code == 404


# ------------------------------------------------------------------- cache API
def test_set_and_get_roundtrip_through_the_gateway(client):
    key = unique("user")
    value = {"name": "John", "tags": ["a", "b"], "n": {"x": 1}}

    r = client.post("/cache", json={"key": key, "value": value, "ttl": 60})
    assert r.status_code == 201
    body = r.json()
    assert body["created"] is True and body["version"] == 1 and body["node_id"] == "node-1"

    got = client.get(path(key)).json()
    assert got["value"] == value and got["version"] == 1
    assert 0 < got["ttl_seconds"] <= 60 and got["node_id"] == "node-1"


def test_overwrite_returns_200_and_bumps_version(client):
    key = unique()
    assert client.post("/cache", json={"key": key, "value": 1}).status_code == 201
    r = client.post("/cache", json={"key": key, "value": 2})
    assert r.status_code == 200
    assert r.json()["created"] is False and r.json()["version"] == 2
    assert client.get(path(key)).json()["value"] == 2


def test_data_really_lives_on_the_node_not_in_the_gateway(client, cluster):
    key = unique()
    node_id = client.post("/cache", json={"key": key, "value": "stored"}).json()["node_id"]
    direct = httpx.get(
        f"{cluster.node_urls[node_id]}/internal/cache/get/{quote(key, safe='')}",
        headers={"X-Cluster-Secret": "test-secret"},
    )
    assert direct.status_code == 200 and direct.json()["value"] == "stored"


def test_get_missing_key_is_404(client):
    r = client.get(path(unique("missing")))
    assert r.status_code == 404 and "not found" in r.json()["detail"]


def test_stored_null_is_a_hit(client):
    key = unique()
    client.post("/cache", json={"key": key, "value": None})
    r = client.get(path(key))
    assert r.status_code == 200 and r.json()["value"] is None


def test_delete_is_idempotent(client):
    key = unique()
    client.post("/cache", json={"key": key, "value": 1})
    first, second = client.delete(path(key)), client.delete(path(key))
    assert (first.status_code, first.json()["deleted"]) == (200, True)
    assert (second.status_code, second.json()["deleted"]) == (200, False)
    assert client.get(path(key)).status_code == 404


def test_exists_and_ttl(client):
    key, forever = unique(), unique()
    assert client.get(f"{path(key)}/exists").json()["exists"] is False
    assert client.get(f"{path(key)}/ttl").json()["state"] == "missing"

    client.post("/cache", json={"key": key, "value": 1, "ttl": 100})
    client.post("/cache", json={"key": forever, "value": 1})
    assert client.get(f"{path(key)}/exists").json()["exists"] is True
    ttl = client.get(f"{path(key)}/ttl").json()
    assert ttl["state"] == "expiring" and 0 < ttl["ttl_seconds"] <= 100
    assert client.get(f"{path(forever)}/ttl").json()["state"] == "no_expiry"


def test_expired_key_is_a_miss(client):
    key = unique()
    client.post("/cache", json={"key": key, "value": 1, "ttl": 0.2})
    time.sleep(0.5)
    assert client.get(path(key)).status_code == 404


@pytest.mark.parametrize("key", ["user:101", "a b", "ünï-çode", "100%", "a?b#c", "k=v&x=1", "  lead"])
def test_keys_with_special_characters_roundtrip(client, key):
    key = f"{key}-{uuid.uuid4().hex[:6]}"
    assert client.post("/cache", json={"key": key, "value": "ok"}).status_code == 201
    assert client.get(path(key)).json()["value"] == "ok"
    assert client.delete(path(key)).json()["deleted"] is True


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize(
    "payload",
    [
        {"key": "", "value": 1},
        {"key": "a/b", "value": 1},  # slash would make the key unreadable via GET
        {"key": "k" * 513, "value": 1},
        {"key": "k"},  # value missing
        {"value": 1},  # key missing
        {"key": "k", "value": 1, "ttl": 0},
        {"key": "k", "value": 1, "ttl": -5},
        {"key": "k", "value": 1, "ttl": "soon"},
        {"key": "k", "value": 1, "TTL": 60},  # misspelled field
    ],
)
def test_invalid_set_requests_are_422(client, payload):
    assert client.post("/cache", json=payload).status_code == 422


def test_overlong_key_in_url_is_422(client):
    assert client.get("/cache/" + "k" * 513).status_code == 422


def test_key_with_slash_in_url_is_not_routable(client):
    assert client.get("/cache/a/b").status_code == 404


# ------------------------------------------------------------- node failures
@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.post("/cache", json={"key": "k", "value": 1}),
        lambda c: c.get("/cache/k"),
        lambda c: c.delete("/cache/k"),
        lambda c: c.get("/cache/k/exists"),
        lambda c: c.get("/cache/k/ttl"),
    ],
)
def test_unreachable_node_gives_503_naming_the_node(dead_node_client, call):
    r = call(dead_node_client)
    assert r.status_code == 503
    assert "node-1" in r.json()["detail"]


def test_gateway_stays_healthy_while_its_node_is_down(dead_node_client):
    assert dead_node_client.get("/health").status_code == 200


@pytest.mark.parametrize(
    "call, expected_attempts",
    [
        (lambda c: c.get("/cache/k"), 3),  # idempotent: 1 try + 2 retries, then stop
        (lambda c: c.get("/cache/k/exists"), 3),
        (lambda c: c.delete("/cache/k"), 3),
        (lambda c: c.post("/cache", json={"key": "k", "value": 1}), 1),  # SET is never retried
    ],
)
def test_slow_node_gives_504_and_retries_are_bounded(hanging_client, hanging_node, call, expected_attempts):
    before = hanging_node.accepted
    r = call(hanging_client)
    time.sleep(0.3)  # let the server thread record the last connection
    assert r.status_code == 504
    assert hanging_node.accepted - before == expected_attempts


def test_secret_mismatch_is_a_502_not_a_cache_miss(wrong_secret_client):
    r = wrong_secret_client.get("/cache/anything")
    assert r.status_code == 502
    assert "401" in r.json()["detail"]
