"""A node that is reachable but SLOW (accepts connections, never answers).

Reads may move to a replica. Writes must NOT: the slow node may still have applied
the write, so retrying elsewhere could create two diverging copies. The client gets
a 504 and decides.
"""
import sys
import uuid
from pathlib import Path
from urllib.parse import quote

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from harness import (  # noqa: E402
    ROOT, SECRET, HangingServer, load_module_from_file, run_gateway, run_node, wait_until,
)

ch = load_module_from_file("gateway_consistent_hash_to", ROOT / "gateway/app/services/consistent_hash.py")
HEADERS = {"X-Cluster-Secret": SECRET}


def key_with_primary(primary):
    ring = ch.HashRing(100)
    ring.add_node("node-1")
    ring.add_node("node-2")
    for i in range(2000):
        key = f"slow-{uuid.uuid4().hex[:6]}-{i}"
        if ring.get_nodes(key, 1)[0] == primary:
            return key
    raise AssertionError("no suitable key found")


def path(key):
    return f"/cache/{quote(key, safe='')}"


@pytest.fixture
def setup():
    hanging = HangingServer()
    with run_node("node-2") as real:
        nodes = {"node-1": f"http://127.0.0.1:{hanging.port}", "node-2": real.url}
        # health tracking effectively off: we are testing per-request behaviour
        with run_gateway(nodes, replication_factor=2, request_timeout=0.3, max_retries=1,
                         health_check_interval=3600, failure_threshold=1000) as gw:
            with httpx.Client(base_url=gw.url, timeout=15) as client:
                yield client, hanging, real
    hanging.close()


def test_reads_fall_back_from_a_slow_primary_to_the_replica(setup):
    client, hanging, real = setup
    key = key_with_primary("node-1")
    # put the value on the real node directly (the hanging node cannot take it)
    r = httpx.post(f"{real.url}/internal/cache/set", headers=HEADERS, json={"key": key, "value": "replica-copy"})
    assert r.status_code in (200, 201)

    got = client.get(path(key))
    assert got.status_code == 200
    assert got.json()["value"] == "replica-copy" and got.json()["fallback"] is True
    assert got.json()["node_id"] == "node-2"


def test_a_write_to_a_slow_primary_is_a_504_and_is_not_redirected(setup):
    client, hanging, real = setup
    key = key_with_primary("node-1")
    before = hanging.accepted

    r = client.post("/cache", json={"key": key, "value": "maybe-applied"})
    wait_until(lambda: hanging.accepted - before >= 1, what="the request to reach the slow node")

    assert r.status_code == 504
    assert hanging.accepted - before == 1  # SET is never retried
    miss = httpx.get(f"{real.url}/internal/cache/get/{quote(key, safe='')}", headers=HEADERS)
    assert miss.status_code == 404  # no failover copy was created
