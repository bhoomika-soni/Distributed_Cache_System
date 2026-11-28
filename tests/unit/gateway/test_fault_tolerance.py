"""Fault-tolerance logic, tested with a FAKE node client: no servers, no waiting.

Covers the node status state machine, routing around unhealthy nodes, the
read-fallback / write-failover rules, the health monitor and re-sync. The same
behaviour is then checked end to end with real processes in
tests/integration/gateway/test_fault_tolerance.py.
"""
import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from harness import ROOT, load_package  # noqa: E402

# Load the gateway's services package under a private name (the top-level `app`
# name is taken by the cache node in the same pytest process).
load_package("gw_services", ROOT / "gateway/app/services")
nm = importlib.import_module("gw_services.node_manager")
nc = importlib.import_module("gw_services.node_client")
nr = importlib.import_module("gw_services.node_router")
cs = importlib.import_module("gw_services.cache_service")
hm = importlib.import_module("gw_services.health_monitor")
rs = importlib.import_module("gw_services.recovery_service")

IDS = ("node-1", "node-2", "node-3")


# ------------------------------------------------------------------- fakes
class FakeClient:
    """Stands in for NodeClient. Nodes in `down` are unreachable, nodes in `slow` time out."""

    def __init__(self, ids=IDS):
        self.down, self.slow = set(), set()
        self.data = {i: {} for i in ids}
        self.calls = []  # (operation, node_id)
        self.set_replicas = []  # (primary, [replica ids])
        self.snapshots = {}  # node_id -> list of entry dicts
        self.applied = []  # (node_id, payload)

    def _enter(self, op, node):
        self.calls.append((op, node.node_id))
        if node.node_id in self.slow:
            raise nc.NodeTimeoutError(node.node_id, f"Node {node.node_id} timed out")
        if node.node_id in self.down:
            raise nc.NodeUnavailableError(node.node_id, f"Node {node.node_id} unavailable")

    def called(self, op=None):
        return [n for o, n in self.calls if op is None or o == op]

    async def set(self, node, key, value, ttl, replicas=()):
        self._enter("set", node)
        self.data[node.node_id][key] = value
        self.set_replicas.append((node.node_id, [r.node_id for r in replicas]))
        return {"key": key, "created": True, "version": 1, "expires_at": None, "ttl_seconds": None}

    async def get(self, node, key):
        self._enter("get", node)
        if key not in self.data[node.node_id]:
            return None
        return {"key": key, "value": self.data[node.node_id][key], "version": 1,
                "expires_at": None, "ttl_seconds": None}

    async def exists(self, node, key):
        self._enter("exists", node)
        return {"key": key, "exists": key in self.data[node.node_id]}

    async def ttl(self, node, key):
        self._enter("ttl", node)
        state = "no_expiry" if key in self.data[node.node_id] else "missing"
        return {"key": key, "state": state, "ttl_seconds": None}

    async def delete(self, node, key, replicas=()):
        self._enter("delete", node)
        existed = self.data[node.node_id].pop(key, None) is not None
        self.set_replicas.append((node.node_id, [r.node_id for r in replicas]))
        return {"key": key, "deleted": existed}

    async def probe(self, node):
        self._enter("probe", node)
        return {"status": "up"}

    async def snapshot(self, node):
        self._enter("snapshot", node)
        return {"entries": self.snapshots.get(node.node_id, [])}

    async def apply(self, node, payload):
        self._enter("apply", node)
        self.applied.append((node.node_id, payload))
        fresh = payload["key"] not in self.data[node.node_id]
        self.data[node.node_id][payload["key"]] = payload["value"]
        return {"applied": fresh, "reason": "applied" if fresh else "stale", "version": payload["version"]}


def make_cluster(ids=IDS, rf=3, failure_threshold=2, recovery_threshold=2):
    nodes = nm.NodeManager(
        [SimpleNamespace(node_id=i, url=f"http://{i}") for i in ids],
        failure_threshold=failure_threshold,
        recovery_threshold=recovery_threshold,
    )
    router = nr.NodeRouter(nodes, virtual_nodes=100, replication_factor=rf)
    client = FakeClient(ids)
    service = cs.CacheService(router, client, nodes)
    return SimpleNamespace(nodes=nodes, router=router, client=client, service=service)


def set_status(cluster, node_id, status):
    cluster.nodes.get(node_id).status = status


def key_with_primary(cluster, node_id):
    for i in range(1000):
        key = f"k{i}"
        if cluster.router.route(key).node_id == node_id:
            return key
    raise AssertionError("no key found")


def run(coro):
    return asyncio.run(coro)


# =================================================== node status state machine
def manager(**kw):
    t = [1000.0]
    m = nm.NodeManager([SimpleNamespace(node_id="a", url="http://a")], clock=lambda: t[0], **kw)
    return m, t


def test_a_new_node_is_unknown_and_its_first_success_makes_it_healthy():
    m, _ = manager()
    assert m.get("a").status == nm.UNKNOWN
    m.record_success("a", probe=True)
    assert m.get("a").status == nm.HEALTHY


def test_a_single_failure_does_not_mark_a_healthy_node_unhealthy():
    m, _ = manager(failure_threshold=3)
    m.record_success("a", probe=True)
    m.record_failure("a", "boom", probe=True)
    m.record_failure("a", "boom", probe=True)
    assert m.get("a").status == nm.HEALTHY  # 2 of 3
    m.record_failure("a", "boom", probe=True)
    assert m.get("a").status == nm.UNHEALTHY


def test_a_success_resets_the_failure_streak():
    m, _ = manager(failure_threshold=2)
    m.record_success("a", probe=True)
    m.record_failure("a", "x")
    m.record_success("a")
    m.record_failure("a", "x")
    assert m.get("a").status == nm.HEALTHY and m.get("a").consecutive_failures == 1


def test_an_unknown_node_that_keeps_failing_becomes_unhealthy():
    m, _ = manager(failure_threshold=2)
    m.record_failure("a", "x", probe=True)
    m.record_failure("a", "x", probe=True)
    assert m.get("a").status == nm.UNHEALTHY


def test_recovery_needs_consecutive_good_probes():
    m, _ = manager(failure_threshold=1, recovery_threshold=2)
    m.record_success("a", probe=True)
    m.record_failure("a", "down", probe=True)
    assert m.get("a").status == nm.UNHEALTHY

    m.record_success("a", probe=True)
    assert m.get("a").status == nm.UNHEALTHY  # 1 of 2
    m.record_failure("a", "flapped", probe=True)  # the streak is broken
    m.record_success("a", probe=True)
    assert m.get("a").status == nm.UNHEALTHY  # back to 1 of 2
    m.record_success("a", probe=True)
    assert m.get("a").status == nm.RECOVERING


def test_stray_request_successes_do_not_revive_an_unhealthy_node():
    m, _ = manager(failure_threshold=1, recovery_threshold=1)
    m.record_success("a", probe=True)
    m.record_failure("a", "down")
    for _ in range(5):
        m.record_success("a")  # not a probe
    assert m.get("a").status == nm.UNHEALTHY


def test_a_failure_while_recovering_sends_the_node_straight_back():
    m, _ = manager(failure_threshold=1, recovery_threshold=1)
    m.record_success("a", probe=True)
    m.record_failure("a", "down", probe=True)
    m.record_success("a", probe=True)
    assert m.get("a").status == nm.RECOVERING
    m.record_failure("a", "down again", probe=True)  # one failure is enough here
    assert m.get("a").status == nm.UNHEALTHY


def test_mark_healthy_only_works_from_recovering():
    m, _ = manager(failure_threshold=1, recovery_threshold=1)
    m.record_success("a", probe=True)
    assert m.mark_healthy("a", {"ok": True}) is False  # already healthy
    m.record_failure("a", "x", probe=True)
    assert m.mark_healthy("a", {"ok": True}) is False  # unhealthy, not recovering
    m.record_success("a", probe=True)
    assert m.mark_healthy("a", {"ok": True, "copied": 3}) is True
    node = m.get("a")
    assert node.status == nm.HEALTHY and node.last_resync["copied"] == 3 and node.last_error is None


def test_node_records_the_error_and_when_its_status_changed():
    m, t = manager(failure_threshold=1)
    m.record_success("a", probe=True)
    t[0] = 1234.0
    m.record_failure("a", "connection refused", probe=True)
    node = m.get("a")
    assert node.last_error == "connection refused"
    assert node.last_status_change_at == 1234.0 and node.last_checked_at == 1234.0


def test_which_nodes_may_serve_reads_and_receive_replicas():
    m = nm.NodeManager([SimpleNamespace(node_id=i, url=i) for i in "abcd"])
    for node_id, status in zip("abcd", (nm.HEALTHY, nm.UNKNOWN, nm.RECOVERING, nm.UNHEALTHY)):
        m.get(node_id).status = status
    assert m.readable_ids() == {"a", "b"}
    assert m.replica_ids() == {"a", "b", "c"}  # recovering nodes still receive writes


# ======================================================================= routing
def ids(nodes):
    return [n.node_id for n in nodes]


def test_read_candidates_follow_ring_order_and_skip_unhealthy_nodes():
    c = make_cluster()
    key = "user:42"
    ring_order = c.router.ring.get_nodes(key, 3)
    assert ids(c.router.read_candidates(key)) == ring_order

    set_status(c, ring_order[0], nm.UNHEALTHY)
    assert ids(c.router.read_candidates(key)) == ring_order[1:]


def test_recovering_nodes_get_no_reads_and_cannot_be_the_write_primary():
    c = make_cluster()
    key = "user:42"
    first = c.router.ring.get_nodes(key, 3)[0]
    set_status(c, first, nm.RECOVERING)
    assert first not in ids(c.router.read_candidates(key))
    assert first not in ids(c.router.acting_primaries(key))


def test_recovering_nodes_do_receive_replicated_writes():
    c = make_cluster()
    key = "user:42"
    order = c.router.ring.get_nodes(key, 3)
    set_status(c, order[1], nm.RECOVERING)
    replicas = ids(c.router.replicas_for(key, c.nodes.get(order[0])))
    assert order[1] in replicas


def test_replicas_exclude_the_primary_and_nodes_that_just_failed():
    c = make_cluster()
    key = "user:42"
    order = c.router.ring.get_nodes(key, 3)
    primary = c.nodes.get(order[0])
    assert ids(c.router.replicas_for(key, primary)) == order[1:]
    assert ids(c.router.replicas_for(key, primary, exclude={order[1]})) == [order[2]]


def test_replication_factor_one_never_has_replicas():
    c = make_cluster(rf=1)
    assert c.router.replicas_for("k", c.router.route("k")) == []


def test_a_down_nodes_keys_move_to_the_next_node_on_the_ring():
    c = make_cluster(ids=("n1", "n2", "n3", "n4", "n5"), rf=3)
    key = "user:42"
    order = c.router.ring.get_nodes(key, 5)
    set_status(c, order[0], nm.UNHEALTHY)
    assert ids(c.router.read_candidates(key)) == order[1:4]  # the next three usable nodes


# ================================================================== reads
def test_read_from_a_healthy_primary_is_not_a_fallback():
    c = make_cluster()
    key = key_with_primary(c, "node-2")
    c.client.data["node-2"][key] = "v"
    result = run(c.service.get(key))
    assert result["value"] == "v" and result["node_id"] == "node-2" and result["fallback"] is False
    assert c.client.called() == ["node-2"]


def test_read_falls_back_to_a_replica_when_the_primary_is_unreachable():
    c = make_cluster()
    key = key_with_primary(c, "node-1")
    for node_id in IDS:
        c.client.data[node_id][key] = "v"
    c.client.down.add("node-1")

    result = run(c.service.get(key))
    assert result["value"] == "v" and result["node_id"] != "node-1" and result["fallback"] is True
    assert c.client.called()[0] == "node-1"  # it was tried first
    assert c.nodes.get("node-1").consecutive_failures == 1  # and the failure was counted


def test_read_falls_back_when_the_primary_is_too_slow_as_well():
    c = make_cluster()
    key = key_with_primary(c, "node-1")
    c.client.data["node-2"][key] = c.client.data["node-3"][key] = "v"
    c.client.slow.add("node-1")
    assert run(c.service.get(key))["fallback"] is True


@pytest.mark.parametrize("op", ["exists", "ttl"])
def test_exists_and_ttl_fall_back_too(op):
    c = make_cluster()
    key = key_with_primary(c, "node-1")
    c.client.data["node-2"][key] = c.client.data["node-3"][key] = "v"
    c.client.down.add("node-1")
    result = run(getattr(c.service, op)(key))
    assert result["fallback"] is True and result["node_id"] != "node-1"


def test_a_cache_miss_is_not_chased_across_replicas():
    """A miss is an answer. Asking every replica would multiply the cost of each miss."""
    c = make_cluster()
    assert run(c.service.get("never-set")) is None
    assert len(c.client.called("get")) == 1


def test_all_candidates_failing_re_raises_the_last_error():
    c = make_cluster()
    c.client.down.update(IDS)
    with pytest.raises(nc.NodeUnavailableError):
        run(c.service.get("k"))
    assert len(c.client.called("get")) == 3  # every candidate was tried exactly once

    c2 = make_cluster()
    c2.client.slow.update(IDS)
    with pytest.raises(nc.NodeTimeoutError):  # still a 504, not disguised as a 503
        run(c2.service.get("k"))


def test_no_usable_node_raises_with_every_nodes_status_in_the_message():
    c = make_cluster()
    for node_id in IDS:
        set_status(c, node_id, nm.UNHEALTHY)
    with pytest.raises(cs.NoAvailableNodesError) as excinfo:
        run(c.service.get("k"))
    assert "node-1=unhealthy" in str(excinfo.value) and "node-3=unhealthy" in str(excinfo.value)
    assert c.client.calls == []  # nothing was even attempted


def test_real_traffic_detects_a_dead_node_and_then_stops_using_it():
    c = make_cluster(failure_threshold=2)
    key = key_with_primary(c, "node-1")
    c.client.data["node-2"][key] = c.client.data["node-3"][key] = "v"
    c.client.down.add("node-1")

    run(c.service.get(key))
    run(c.service.get(key))
    assert c.nodes.get("node-1").status == nm.UNHEALTHY  # two failed requests were enough

    c.client.calls.clear()
    run(c.service.get(key))
    assert "node-1" not in c.client.called()  # no more time wasted on the dead node


def test_a_successful_request_resets_the_failure_count():
    c = make_cluster()
    key = key_with_primary(c, "node-1")
    c.client.data["node-1"][key] = "v"
    c.client.down.add("node-1")
    run(c.service.get(key))
    c.client.down.clear()
    run(c.service.get(key))
    assert c.nodes.get("node-1").consecutive_failures == 0


# ================================================================== writes
def test_normal_write_goes_to_the_ring_primary_with_the_next_nodes_as_replicas():
    c = make_cluster()
    key = "user:42"
    order = c.router.ring.get_nodes(key, 3)
    result = run(c.service.set(key, "v", None))
    assert result["node_id"] == order[0] and result["replicas"] == order[1:]
    assert result["fallback"] is False and result["replication"] == "async"
    assert c.client.set_replicas == [(order[0], order[1:])]


def test_write_fails_over_to_the_next_node_when_the_primary_is_unreachable():
    c = make_cluster()
    key = "user:42"
    order = c.router.ring.get_nodes(key, 3)
    c.client.down.add(order[0])

    result = run(c.service.set(key, "v", None))
    assert result["node_id"] == order[1] and result["fallback"] is True
    assert result["replicas"] == [order[2]]  # the failed primary is NOT a replica target
    assert c.nodes.get(order[0]).consecutive_failures == 1
    assert key in c.client.data[order[1]] and key not in c.client.data[order[0]]


def test_a_slow_primary_is_never_failed_over_because_the_write_may_have_happened():
    c = make_cluster()
    key = "user:42"
    order = c.router.ring.get_nodes(key, 3)
    c.client.slow.add(order[0])

    with pytest.raises(nc.NodeTimeoutError):
        run(c.service.set(key, "v", None))
    assert c.client.called("set") == [order[0]]  # no other node was asked to store it
    assert all(key not in data for data in c.client.data.values())
    assert c.nodes.get(order[0]).consecutive_failures == 1  # but the slowness is counted


def test_delete_fails_over_like_set():
    c = make_cluster()
    key = "user:42"
    order = c.router.ring.get_nodes(key, 3)
    c.client.data[order[1]][key] = "v"
    c.client.down.add(order[0])
    result = run(c.service.delete(key))
    assert result["node_id"] == order[1] and result["deleted"] is True and result["fallback"] is True


def test_write_when_every_node_is_unreachable_raises_the_last_error():
    c = make_cluster()
    c.client.down.update(IDS)
    with pytest.raises(nc.NodeUnavailableError):
        run(c.service.set("k", "v", None))
    assert len(c.client.called("set")) == 3


def test_write_with_no_usable_node_raises_no_available_nodes():
    c = make_cluster()
    for node_id in IDS:
        set_status(c, node_id, nm.UNHEALTHY)
    with pytest.raises(cs.NoAvailableNodesError):
        run(c.service.set("k", "v", None))


def test_a_recovering_node_is_never_the_write_primary_but_does_get_the_replica_copy():
    c = make_cluster()
    key = "user:42"
    order = c.router.ring.get_nodes(key, 3)
    set_status(c, order[0], nm.RECOVERING)
    result = run(c.service.set(key, "v", None))
    assert result["node_id"] == order[1]  # its version numbers may be behind: never primary
    assert order[0] in result["replicas"]  # but it catches up through replication


def test_with_one_node_left_writes_still_work_without_replicas():
    c = make_cluster()
    set_status(c, "node-1", nm.UNHEALTHY)
    set_status(c, "node-2", nm.UNHEALTHY)
    result = run(c.service.set("k", "v", None))
    assert result["node_id"] == "node-3" and result["replicas"] == [] and result["replication"] == "none"


# =============================================================== health monitor
class FakeRecovery:
    def __init__(self):
        self.started = []

    def start(self, node_id):
        self.started.append(node_id)


def make_monitor(c, interval=0.01):
    recovery = FakeRecovery()
    return hm.HealthMonitor(c.nodes, c.client, recovery, interval), recovery


def test_probes_mark_reachable_nodes_healthy_and_dead_nodes_unhealthy_after_the_threshold():
    c = make_cluster(failure_threshold=2)
    c.client.down.add("node-2")
    monitor, _ = make_monitor(c)

    run(monitor.check_all())
    assert c.nodes.get("node-1").status == nm.HEALTHY and c.nodes.get("node-3").status == nm.HEALTHY
    assert c.nodes.get("node-2").status == nm.UNKNOWN  # one failure is not enough

    run(monitor.check_all())
    assert c.nodes.get("node-2").status == nm.UNHEALTHY
    assert c.nodes.get("node-2").last_checked_at is not None


def test_a_node_that_comes_back_becomes_recovering_and_its_resync_is_started():
    async def scenario():
        c = make_cluster(failure_threshold=1, recovery_threshold=2)
        monitor, recovery = make_monitor(c)
        c.client.down.add("node-2")
        await monitor.check_all()
        assert c.nodes.get("node-2").status == nm.UNHEALTHY

        c.client.down.clear()
        await monitor.check_all()
        assert c.nodes.get("node-2").status == nm.UNHEALTHY and recovery.started == []
        await monitor.check_all()
        assert c.nodes.get("node-2").status == nm.RECOVERING
        assert recovery.started == ["node-2"]

    run(scenario())


def test_any_probe_error_counts_against_the_node_not_just_known_ones():
    c = make_cluster(failure_threshold=1)

    async def weird_probe(node):
        raise RuntimeError("unexpected")

    c.client.probe = weird_probe
    monitor, _ = make_monitor(c)
    run(monitor.check_all())
    assert all(n.status == nm.UNHEALTHY for n in c.nodes.all())
    assert c.nodes.get("node-1").last_error == "unexpected"


def test_the_monitor_loop_runs_repeatedly_and_stops_cleanly():
    async def scenario():
        c = make_cluster()
        monitor, _ = make_monitor(c, interval=0.01)
        monitor.start()
        monitor.start()  # second start is a no-op
        await asyncio.sleep(0.1)
        assert monitor.is_running and len(c.client.called("probe")) >= 6  # several rounds
        await monitor.stop()
        await monitor.stop()
        assert not monitor.is_running

    run(scenario())


def test_the_monitor_survives_a_failing_round():
    async def scenario():
        c = make_cluster()
        monitor, _ = make_monitor(c, interval=0.01)
        rounds = []
        original = monitor.check_all

        async def flaky():
            rounds.append(1)
            if len(rounds) == 1:
                raise RuntimeError("boom")
            await original()

        monitor.check_all = flaky
        monitor.start()
        await asyncio.sleep(0.1)
        still_running = monitor.is_running
        await monitor.stop()
        return still_running, len(rounds)

    still_running, rounds = run(scenario())
    assert still_running and rounds >= 2


# ====================================================================== re-sync
def entry(key, value="v", version=1, expires_at=None):
    return {"key": key, "value": value, "version": version, "created_at": 1.0, "expires_at": expires_at}


def recovering_cluster(rf=2):
    c = make_cluster(rf=rf)
    for node_id in ("node-1", "node-2"):
        set_status(c, node_id, nm.HEALTHY)
    set_status(c, "node-3", nm.RECOVERING)
    c.recovery = rs.RecoveryService(c.nodes, c.router, c.client)
    return c


def test_resync_copies_only_the_entries_the_node_should_hold():
    async def scenario():
        c = recovering_cluster(rf=2)
        entries = [entry(f"key-{i}", value=i, version=i + 1, expires_at=5000.0 + i) for i in range(40)]
        c.client.snapshots["node-1"] = entries
        c.client.snapshots["node-2"] = entries  # replicas hold the same data

        result = await c.recovery.resync(c.nodes.get("node-3"))
        return c, entries, result

    c, entries, result = run(scenario())
    should_hold = {e["key"] for e in entries if "node-3" in c.router.ring.get_nodes(e["key"], 2)}
    assert 0 < len(should_hold) < 40  # the ownership filter really filtered

    sent = {p["key"]: p for n, p in c.client.applied if n == "node-3"}
    assert set(sent) == should_hold
    assert result["copied"] == len(should_hold)
    assert result["already_current"] == len(should_hold)  # the second source's copies were skipped
    assert result["sources"] == ["node-1", "node-2"]

    sample = sent[sorted(should_hold)[0]]
    assert sample["op"] == "set" and sample["origin"] == "gateway-resync"
    original = next(e for e in entries if e["key"] == sample["key"])
    assert sample["version"] == original["version"] and sample["expires_at"] == original["expires_at"]


def test_a_successful_resync_puts_the_node_back_into_service():
    async def scenario():
        c = recovering_cluster(rf=3)
        c.client.snapshots["node-1"] = [entry("a"), entry("b")]
        c.recovery.start("node-3")
        c.recovery.start("node-3")  # no second task while one is running
        assert c.recovery.is_running("node-3")
        await c.recovery.wait_idle()
        return c

    c = run(scenario())
    node = c.nodes.get("node-3")
    assert node.status == nm.HEALTHY
    assert node.last_resync["ok"] is True and node.last_resync["copied"] == 2
    assert not c.recovery.is_running("node-3")


def test_unhealthy_peers_are_not_used_as_a_source():
    async def scenario():
        c = recovering_cluster(rf=3)
        set_status(c, "node-2", nm.UNHEALTHY)
        c.client.snapshots["node-1"] = [entry("a")]
        c.client.snapshots["node-2"] = [entry("zzz")]
        return c, await c.recovery.resync(c.nodes.get("node-3"))

    c, result = run(scenario())
    assert result["sources"] == ["node-1"]
    assert "snapshot" not in [op for op, n in c.client.calls if n == "node-2"]


def test_with_no_healthy_peer_there_is_nothing_to_copy_and_the_node_goes_live():
    async def scenario():
        c = make_cluster()
        set_status(c, "node-1", nm.UNHEALTHY)
        set_status(c, "node-2", nm.UNHEALTHY)
        set_status(c, "node-3", nm.RECOVERING)
        recovery = rs.RecoveryService(c.nodes, c.router, c.client)
        recovery.start("node-3")
        await recovery.wait_idle()
        return c

    c = run(scenario())
    assert c.nodes.get("node-3").status == nm.HEALTHY
    assert c.nodes.get("node-3").last_resync["copied"] == 0


def test_a_failed_resync_leaves_the_node_recovering_and_is_recorded_then_retried():
    async def scenario():
        c = recovering_cluster(rf=3)
        c.client.snapshots["node-1"] = [entry("a")]
        c.client.down.add("node-1")  # the source dies mid-recovery
        c.recovery.start("node-3")
        await c.recovery.wait_idle()
        first = (c.nodes.get("node-3").status, dict(c.nodes.get("node-3").last_resync))

        c.client.down.clear()  # next health-monitor tick starts another attempt
        c.recovery.start("node-3")
        await c.recovery.wait_idle()
        return c, first

    c, (status, failure) = run(scenario())
    assert status == nm.RECOVERING  # NOT put into service half-synced
    assert failure["ok"] is False and "node-1" in failure["error"] and failure["attempt"] == 1
    assert c.nodes.get("node-3").status == nm.HEALTHY  # the retry succeeded


def test_the_target_failing_during_resync_keeps_it_out_of_service():
    async def scenario():
        c = recovering_cluster(rf=3)
        c.client.snapshots["node-1"] = [entry("a")]
        c.client.down.add("node-3")
        c.recovery.start("node-3")
        await c.recovery.wait_idle()
        return c

    c = run(scenario())
    assert c.nodes.get("node-3").status == nm.RECOVERING
    assert c.nodes.get("node-3").resync_attempts == 1


def test_resync_aborts_if_the_node_fails_again_before_it_finishes():
    c = recovering_cluster(rf=3)
    set_status(c, "node-3", nm.UNHEALTHY)  # the monitor sent it back
    with pytest.raises(rs.ResyncAborted):
        run(c.recovery.resync(c.nodes.get("node-3")))


def test_stop_cancels_running_resyncs():
    async def scenario():
        c = recovering_cluster(rf=3)

        async def slow_snapshot(node):
            await asyncio.sleep(60)

        c.client.snapshot = slow_snapshot
        c.recovery.start("node-3")
        await asyncio.sleep(0.02)
        assert c.recovery.is_running("node-3")
        await c.recovery.stop()
        return c

    c = run(scenario())
    assert not c.recovery.is_running("node-3")
    assert c.nodes.get("node-3").status == nm.RECOVERING  # cancelled, not "failed"
