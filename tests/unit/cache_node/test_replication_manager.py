"""ReplicationManager against a fake network (httpx.MockTransport): no servers, no real waiting."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from app.replication.replication_manager import ReplicationManager

PAYLOAD = {"op": "set", "key": "k", "value": 1, "version": 1, "origin": "node-1", "sent_at": 0.0}
REPLICA_2 = SimpleNamespace(node_id="node-2", url="http://node-2")
REPLICA_3 = SimpleNamespace(node_id="node-3", url="http://node-3")


def run(handler, scenario, max_retries=2):
    """Run `scenario(manager)` against a fake network; return (manager, requests)."""
    requests = []

    def recording(request):
        requests.append(request)
        return handler(request, len(requests))

    async def main():
        async with httpx.AsyncClient(transport=httpx.MockTransport(recording)) as http:
            manager = ReplicationManager("node-1", http, "secret", max_retries=max_retries,
                                         backoff_base=0.0)
            await scenario(manager)
            return manager

    return asyncio.run(main()), requests


def ok(request, n):
    return httpx.Response(200, json={"applied": True, "reason": "applied", "version": 1})


async def send_and_wait(manager, targets=(REPLICA_2,)):
    scheduled = manager.replicate(PAYLOAD, targets)
    await manager.drain()
    return scheduled


# --------------------------------------------------------------------- delivery
def test_successful_delivery_posts_the_payload_with_the_secret():
    manager, requests = run(ok, send_and_wait)
    assert len(requests) == 1
    assert requests[0].url == "http://node-2/internal/replication/apply"
    assert requests[0].headers["x-cluster-secret"] == "secret"
    assert json.loads(requests[0].content) == PAYLOAD
    s = manager.stats
    assert (s.scheduled, s.succeeded, s.failed, s.retries) == (1, 1, 0, 0)
    assert manager.in_flight == 0


def test_replicate_returns_immediately_without_waiting_for_delivery():
    """Asynchronous: the caller (the client's request) must not wait for replicas."""
    async def scenario(manager):
        scheduled = manager.replicate(PAYLOAD, [REPLICA_2, REPLICA_3])
        assert scheduled == ["node-2", "node-3"]
        assert manager.in_flight == 2  # still pending: nothing has been sent yet
        await manager.drain()

    manager, requests = run(ok, scenario)
    assert len(requests) == 2 and manager.stats.succeeded == 2


def test_one_failing_replica_does_not_affect_the_others():
    def handler(request, n):
        if request.url.host == "node-3":
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={})

    async def scenario(manager):
        await send_and_wait(manager, [REPLICA_2, REPLICA_3])

    manager, _ = run(handler, scenario, max_retries=1)
    s = manager.stats
    assert (s.scheduled, s.succeeded, s.failed) == (2, 1, 1)
    assert manager.recent_failures[0]["target"] == "node-3"


def test_a_replica_that_ignores_the_write_as_stale_still_counts_as_success():
    stale = lambda r, n: httpx.Response(200, json={"applied": False, "reason": "stale"})
    manager, _ = run(stale, send_and_wait)
    assert manager.stats.succeeded == 1 and manager.stats.failed == 0


# ----------------------------------------------------------- bounded retries
def test_retries_are_bounded_then_the_failure_is_recorded_not_dropped():
    def refuse(request, n):
        raise httpx.ConnectError("refused", request=request)

    manager, requests = run(refuse, send_and_wait, max_retries=2)
    assert len(requests) == 3  # 1 try + 2 retries, then it stops
    s = manager.stats
    assert (s.scheduled, s.succeeded, s.failed, s.retries) == (1, 0, 1, 2)

    failure = manager.recent_failures[0]
    assert (failure["op"], failure["key"], failure["version"]) == ("set", "k", 1)
    assert failure["target"] == "node-2" and failure["attempts"] == 3
    assert "ConnectError" in failure["error"]


def test_delivery_succeeds_if_the_replica_comes_back_during_retries():
    def flaky(request, n):
        if n < 3:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={})

    manager, requests = run(flaky, send_and_wait, max_retries=2)
    assert len(requests) == 3
    s = manager.stats
    assert (s.succeeded, s.failed, s.retries) == (1, 0, 2)


def test_server_errors_are_retried_but_client_errors_are_not():
    manager, requests = run(lambda r, n: httpx.Response(503), send_and_wait, max_retries=2)
    assert len(requests) == 3 and manager.stats.failed == 1

    manager, requests = run(lambda r, n: httpx.Response(401), send_and_wait, max_retries=2)
    assert len(requests) == 1 and manager.stats.failed == 1  # wrong secret will not fix itself
    assert manager.recent_failures[0]["error"] == "HTTP 401"


# ------------------------------------------------------------ loop prevention
def test_a_node_never_replicates_to_itself():
    node_1 = SimpleNamespace(node_id="node-1", url="http://node-1")

    async def scenario(manager):
        assert manager.replicate(PAYLOAD, [node_1, REPLICA_2]) == ["node-2"]
        await manager.drain()

    manager, requests = run(ok, scenario)
    assert [r.url.host for r in requests] == ["node-2"]
    assert manager.stats.scheduled == 1


# ----------------------------------------------------------- receiving side
@pytest.mark.parametrize(
    "reason, field_name",
    [("applied", "applied"), ("stale", "ignored_stale"), ("expired", "ignored_expired")],
)
def test_record_received_counts_each_outcome_and_the_lag(reason, field_name):
    async def scenario(manager):
        manager.clock = lambda: 105.0
        manager.record_received(reason, sent_at=100.0)

    manager, _ = run(ok, scenario)
    s = manager.stats
    assert s.received == 1 and getattr(s, field_name) == 1
    assert s.last_lag_seconds == pytest.approx(5.0)


# ---------------------------------------------------------------- shutdown
def test_close_cancels_stuck_deliveries_and_records_them_as_failures():
    async def scenario(manager):
        manager.replicate(PAYLOAD, [REPLICA_2])
        await manager.close(timeout=0.05)
        assert manager.in_flight == 0

    # A transport that never answers, so the delivery is still in flight at shutdown.
    async def never_answers(request):
        await asyncio.sleep(60)

    async def main():
        transport = httpx.MockTransport(never_answers)
        async with httpx.AsyncClient(transport=transport) as http:
            manager = ReplicationManager("node-1", http, "secret", backoff_base=0.0)
            await scenario(manager)
            return manager

    manager = asyncio.run(main())
    assert manager.stats.failed == 1
    assert "shutdown" in manager.recent_failures[0]["error"]


def test_snapshot_has_the_documented_shape():
    manager, _ = run(ok, send_and_wait)
    snap = manager.snapshot()
    assert set(snap) == {"node_id", "sent", "received", "recent_failures"}
    assert snap["sent"]["scheduled"] == 1 and snap["sent"]["in_flight"] == 0
