"""Test helpers: start REAL node and gateway processes on free local ports.

Why processes instead of importing the apps?
  gateway/ and cache_node/ both have a package named `app`; two different
  packages with the same name cannot be imported into one Python process.
  Running each service as its own process (exactly how it runs in production)
  avoids that, tests real HTTP, and lets later phases kill a node to test
  failure handling.
"""
from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
SECRET = "test-secret"
LOCALHOST = "127.0.0.1"


def free_port() -> int:
    """Ask the OS for an unused port (small race window, fine for tests)."""
    with socket.socket() as s:
        s.bind((LOCALHOST, 0))
        return s.getsockname()[1]


def load_module_from_file(name: str, path: Path):
    """Import one file under a unique name (used for modules inside a clashing `app` package)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # must be registered before exec (dataclasses etc.)
    spec.loader.exec_module(module)
    return module


def load_package(name: str, directory: Path):
    """Import a directory as a package called `name` (relative imports inside it work).
    Used to test gateway/app/services without importing the clashing top-level `app`."""
    spec = importlib.util.spec_from_file_location(
        name, directory / "__init__.py", submodule_search_locations=[str(directory)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Service:
    """A `python -m app.main` child process with its output captured for debugging."""

    def __init__(self, name: str, cwd: Path, env: dict[str, str]) -> None:
        self.name = name
        self._log = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "app.main"],
            cwd=cwd,
            env={**os.environ, "PYTHONUNBUFFERED": "1", **env},
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )

    def logs(self) -> str:
        self._log.seek(0)
        return self._log.read().decode(errors="replace")

    def wait_ready(self, url: str, headers: dict | None = None, timeout: float = 25.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"{self.name} exited with code {self._proc.returncode}:\n{self.logs()}")
            try:
                if httpx.get(url, headers=headers, timeout=0.5).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        raise RuntimeError(f"{self.name} not ready after {timeout}s:\n{self.logs()}")

    def stop(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._log.close()


@dataclass
class Running:
    url: str
    service: Service


@contextmanager
def run_node(node_id: str, secret: str = SECRET, capacity: int = 1000, port: int | None = None):
    port = port or free_port()
    svc = Service(node_id, ROOT / "cache_node", {
        "NODE_ID": node_id, "NODE_HOST": LOCALHOST, "NODE_PORT": str(port),
        "CLUSTER_SECRET": secret, "CACHE_CAPACITY": str(capacity),
        "TTL_CLEANUP_INTERVAL": "0.2",
        # short timeout so a dead replica is detected quickly (even on Windows)
        "REPLICATION_TIMEOUT": "1.0", "REPLICATION_RETRIES": "2",
    })
    try:
        url = f"http://{LOCALHOST}:{port}"
        svc.wait_ready(f"{url}/internal/health", {"X-Cluster-Secret": secret})
        yield Running(url, svc)
    finally:
        svc.stop()


@contextmanager
def run_gateway(
    nodes: dict[str, str],
    secret: str = SECRET,
    request_timeout: float = 2.0,
    max_retries: int = 2,
    replication_factor: int = 1,
    health_check_interval: float = 0.3,  # fast, so tests see failures within about a second
    failure_threshold: int = 2,
    recovery_threshold: int = 2,
):
    port = free_port()
    svc = Service("gateway", ROOT / "gateway", {
        "GATEWAY_HOST": LOCALHOST, "GATEWAY_PORT": str(port),
        "NODES": ",".join(f"{i}={u}" for i, u in nodes.items()),
        "CLUSTER_SECRET": secret, "REPLICATION_FACTOR": str(replication_factor),
        "REQUEST_TIMEOUT": str(request_timeout), "MAX_RETRIES": str(max_retries),
        "HEALTH_CHECK_INTERVAL": str(health_check_interval),
        "FAILURE_THRESHOLD": str(failure_threshold),
        "RECOVERY_THRESHOLD": str(recovery_threshold),
    })
    try:
        url = f"http://{LOCALHOST}:{port}"
        svc.wait_ready(f"{url}/health")
        yield Running(url, svc)
    finally:
        svc.stop()


@dataclass
class Cluster:
    gateway_url: str
    node_urls: dict[str, str]
    node_services: dict[str, Service]
    _stack: ExitStack

    def stop_node(self, node_id: str) -> None:
        """Kill a node process (a crash: its memory is gone)."""
        self.node_services[node_id].stop()

    def restart_node(self, node_id: str) -> None:
        """Start a FRESH (empty) node on the same address as the stopped one."""
        self.node_services[node_id].stop()  # harmless if already stopped
        port = int(self.node_urls[node_id].rsplit(":", 1)[1])
        node = self._stack.enter_context(run_node(node_id, port=port))
        self.node_services[node_id] = node.service


@contextmanager
def running_cluster(node_count: int = 2, replication_factor: int = 1, **gateway_kwargs):
    """N real cache nodes plus one gateway in front of them.
    Extra keyword arguments are passed to run_gateway()."""
    with ExitStack() as stack:
        node_urls, node_services = {}, {}
        for i in range(1, node_count + 1):
            node = stack.enter_context(run_node(f"node-{i}"))
            node_urls[f"node-{i}"] = node.url
            node_services[f"node-{i}"] = node.service
        gateway = stack.enter_context(
            run_gateway(node_urls, replication_factor=replication_factor, **gateway_kwargs)
        )
        yield Cluster(gateway.url, node_urls, node_services, stack)


class HangingServer:
    """Accepts TCP connections and never answers - simulates a stuck node.

    `accepted` counts connections, which lets tests prove exactly how many
    attempts the gateway made (i.e. that retries are bounded).
    """

    def __init__(self) -> None:
        self._sock = socket.socket()
        self._sock.bind((LOCALHOST, 0))
        self._sock.listen(16)
        self._sock.settimeout(0.1)
        self.port = self._sock.getsockname()[1]
        self._conns: list[socket.socket] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self._conns.append(conn)

    @property
    def accepted(self) -> int:
        return len(self._conns)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        for c in self._conns:
            c.close()
        self._sock.close()


def wait_until(predicate, timeout: float = 10.0, interval: float = 0.05, what: str = "condition"):
    """Poll until `predicate()` is truthy. Replication is asynchronous, so tests
    wait for the cluster to converge instead of sleeping a fixed time."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for: {what}")
        time.sleep(interval)
