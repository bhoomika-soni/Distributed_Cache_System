"""Unit tests for the gateway's NodeClient retry rules.

Uses httpx.MockTransport (a fake network), so no servers and no real waiting.
These rules decide how the gateway behaves when a node is slow or down, so
they are tested precisely, including the cases where we must NOT retry.
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from harness import ROOT, load_module_from_file  # noqa: E402

nc = load_module_from_file("gateway_node_client", ROOT / "gateway/app/services/node_client.py")

NODE = SimpleNamespace(node_id="node-1", url="http://node-1")


def run_client(handler, operation, max_retries=2):
    """Run `operation(client)` against a fake network; return (result_or_exc, attempts)."""
    attempts = []

    def counting_handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return handler(request, len(attempts))

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(counting_handler)) as http:
            client = nc.NodeClient(http, "secret", max_retries=max_retries, backoff_base=0.0)
            try:
                return await operation(client)
            except nc.NodeError as exc:
                return exc

    return asyncio.run(scenario()), attempts


def refuse(request, n):
    raise httpx.ConnectError("refused", request=request)


def read_timeout(request, n):
    raise httpx.ReadTimeout("slow", request=request)


# ------------------------------------------------------------ happy paths
def test_get_hit_returns_json_and_sends_the_secret_header():
    result, attempts = run_client(
        lambda r, n: httpx.Response(200, json={"key": "k", "value": 1}),
        lambda c: c.get(NODE, "k"),
    )
    assert result == {"key": "k", "value": 1}
    assert attempts[0].headers["x-cluster-secret"] == "secret"


def test_get_404_is_a_miss_not_an_error():
    result, attempts = run_client(lambda r, n: httpx.Response(404, json={"detail": "x"}),
                                  lambda c: c.get(NODE, "k"))
    assert result is None
    assert len(attempts) == 1  # a miss is never retried


def test_keys_are_percent_encoded_in_the_url():
    _, attempts = run_client(lambda r, n: httpx.Response(200, json={}),
                             lambda c: c.exists(NODE, "a b?c#d%"))
    assert attempts[0].url.raw_path == b"/internal/cache/exists/a%20b%3Fc%23d%25"


# ------------------------------------------------------------ bounded retries
def test_connection_refused_is_retried_but_bounded():
    result, attempts = run_client(refuse, lambda c: c.get(NODE, "k"), max_retries=2)
    assert isinstance(result, nc.NodeUnavailableError) and not isinstance(result, nc.NodeTimeoutError)
    assert len(attempts) == 3  # 1 try + 2 retries, then it gives up
    assert "node-1" in str(result)


def test_connect_timeout_is_unavailable_not_a_timeout_and_is_retried():
    """Windows reports a closed port as a connect TIMEOUT. We never reached the
    node, so this must be 503 (unavailable), not 504 (node too slow)."""
    def connect_timeout(request, n):
        raise httpx.ConnectTimeout("no route", request=request)

    result, attempts = run_client(connect_timeout, lambda c: c.get(NODE, "k"), max_retries=2)
    assert isinstance(result, nc.NodeUnavailableError)
    assert not isinstance(result, nc.NodeTimeoutError)
    assert len(attempts) == 3


def test_zero_retries_means_exactly_one_attempt():
    _, attempts = run_client(refuse, lambda c: c.get(NODE, "k"), max_retries=0)
    assert len(attempts) == 1


def test_retry_succeeds_if_the_node_comes_back():
    def flaky(request, n):
        if n < 3:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"key": "k", "value": "ok"})

    result, attempts = run_client(flaky, lambda c: c.get(NODE, "k"), max_retries=2)
    assert result["value"] == "ok" and len(attempts) == 3


def test_connection_refused_on_set_is_retried_because_nothing_was_sent():
    _, attempts = run_client(refuse, lambda c: c.set(NODE, "k", 1, None), max_retries=2)
    assert len(attempts) == 3


@pytest.mark.parametrize("op", ["get", "exists", "ttl", "delete"])
def test_read_timeout_is_retried_for_idempotent_operations(op):
    result, attempts = run_client(read_timeout, lambda c: getattr(c, op)(NODE, "k"), max_retries=2)
    assert isinstance(result, nc.NodeTimeoutError)
    assert len(attempts) == 3


def test_read_timeout_on_set_is_NOT_retried():
    """The first SET may already be applied; repeating it would bump the version twice."""
    result, attempts = run_client(read_timeout, lambda c: c.set(NODE, "k", 1, None), max_retries=2)
    assert isinstance(result, nc.NodeTimeoutError)
    assert len(attempts) == 1


# ------------------------------------------------- HTTP errors are never retried
@pytest.mark.parametrize("status", [401, 500, 503])
def test_error_responses_are_not_retried_and_become_response_errors(status):
    result, attempts = run_client(lambda r, n: httpx.Response(status, json={"detail": "x"}),
                                  lambda c: c.get(NODE, "k"))
    assert isinstance(result, nc.NodeResponseError) and result.status_code == status
    assert len(attempts) == 1


def test_non_json_success_body_is_a_response_error():
    result, _ = run_client(lambda r, n: httpx.Response(200, text="<html>oops</html>"),
                           lambda c: c.exists(NODE, "k"))
    assert isinstance(result, nc.NodeResponseError)
