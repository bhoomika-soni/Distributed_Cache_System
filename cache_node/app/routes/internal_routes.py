"""Internal API of a cache node. Only the gateway (and peer nodes, for
replication) should call these; every route requires the X-Cluster-Secret header.

IMPORTANT: handlers are `async def` on purpose. FastAPI runs plain `def`
handlers in a thread pool, and then two requests could touch the engine at the
same time. The engine has no locks because it relies on everything running on
the single event-loop thread. Do not turn these into `def`.

Keys are captured with `{key:path}` so keys containing "/" (e.g. "a/b") work.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.cache.cache_engine import TTL_MISSING, TTL_NO_EXPIRY, CacheEngine
from app.config import settings
from app.deps import get_engine, get_replication
from app.replication.replication_manager import ReplicationManager
from app.schemas import (
    ApplyRequest,
    ApplyResponse,
    DeleteRequest,
    DeleteResponse,
    ExistsResponse,
    GetResponse,
    SetRequest,
    SetResponse,
    TTLResponse,
)
from app.security import require_cluster_secret

# Every route on this router requires the cluster secret.
router = APIRouter(
    prefix="/internal",
    tags=["internal"],
    dependencies=[Depends(require_cluster_secret)],
)


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


@router.get("/health")
async def internal_health(request: Request, engine: CacheEngine = Depends(get_engine)) -> dict:
    """Node health, polled by the gateway's health monitor (Phase 7)."""
    return {
        "status": "up",
        "node_id": settings.node_id,
        "host": settings.host,
        "port": settings.port,
        "capacity": settings.capacity,
        "keys": engine.size(),
        "uptime_seconds": round(time.time() - request.app.state.started_at, 2),
    }


# ------------------------------------------------------------ client operations
@router.post("/cache/set", response_model=SetResponse)
async def set_key(
    body: SetRequest,
    engine: CacheEngine = Depends(get_engine),
    replication: ReplicationManager = Depends(get_replication),
) -> SetResponse:
    entry = engine.set(body.key, body.value, body.ttl)

    scheduled: list[str] = []
    if body.replicas:
        # Replicate the ABSOLUTE expiry (expires_at) and the version, not the TTL.
        # This only SCHEDULES delivery; we do not wait for the replicas.
        scheduled = replication.replicate(
            {
                "op": "set",
                "key": entry.key,
                "value": entry.value,
                "version": entry.version,
                "created_at": entry.created_at,
                "expires_at": entry.expires_at,
                "origin": settings.node_id,
                "sent_at": time.time(),
            },
            body.replicas,
        )
    return SetResponse(
        key=entry.key,
        created=entry.version == 1,
        version=entry.version,
        expires_at=entry.expires_at,
        ttl_seconds=_round(entry.ttl_remaining(time.time())),
        replicas_scheduled=scheduled,
    )


@router.get("/cache/get/{key:path}", response_model=GetResponse)
async def get_key(key: str, engine: CacheEngine = Depends(get_engine)) -> GetResponse:
    entry = engine.get(key)
    if entry is None:
        # A miss is a normal outcome, but 404 lets the gateway tell
        # "not cached" apart from "node unreachable" without parsing bodies.
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Key '{key}' not found")
    return GetResponse(
        key=entry.key,
        value=entry.value,
        version=entry.version,
        created_at=entry.created_at,
        last_accessed_at=entry.last_accessed_at,
        expires_at=entry.expires_at,
        ttl_seconds=_round(entry.ttl_remaining(time.time())),
    )


@router.delete("/cache/delete/{key:path}", response_model=DeleteResponse)
async def delete_key(
    key: str,
    body: DeleteRequest | None = None,
    engine: CacheEngine = Depends(get_engine),
    replication: ReplicationManager = Depends(get_replication),
) -> DeleteResponse:
    # Idempotent: deleting a missing key is 200 with deleted=false, not an
    # error. Retries can safely repeat a delete.
    version = engine.delete_with_version(key)

    scheduled: list[str] = []
    if version is not None and body is not None and body.replicas:
        # Only a delete that really removed something is replicated. The
        # delete carries its own version so replicas can tell it apart from an
        # older SET that arrives late.
        scheduled = replication.replicate(
            {
                "op": "delete",
                "key": key,
                "version": version,
                "origin": settings.node_id,
                "sent_at": time.time(),
            },
            body.replicas,
        )
    return DeleteResponse(key=key, deleted=version is not None, replicas_scheduled=scheduled)


@router.get("/cache/exists/{key:path}", response_model=ExistsResponse)
async def exists_key(key: str, engine: CacheEngine = Depends(get_engine)) -> ExistsResponse:
    return ExistsResponse(key=key, exists=engine.exists(key))


@router.get("/cache/ttl/{key:path}", response_model=TTLResponse)
async def ttl_key(key: str, engine: CacheEngine = Depends(get_engine)) -> TTLResponse:
    remaining = engine.ttl(key)
    if remaining == TTL_MISSING:
        return TTLResponse(key=key, state="missing", ttl_seconds=None)
    if remaining == TTL_NO_EXPIRY:
        return TTLResponse(key=key, state="no_expiry", ttl_seconds=None)
    return TTLResponse(key=key, state="expiring", ttl_seconds=_round(remaining))


# ---------------------------------------------------------------- replication
@router.post("/replication/apply", response_model=ApplyResponse)
async def apply_replicated_write(
    body: ApplyRequest,
    engine: CacheEngine = Depends(get_engine),
    replication: ReplicationManager = Depends(get_replication),
) -> ApplyResponse:
    """Receive a write replicated from a primary.

    LOOP PREVENTION: this handler only applies the write locally. It never
    calls `replication.replicate()`, so a replicated write cannot be forwarded
    again - it travels exactly one hop.

    200 is returned both when the write was applied and when it was ignored as
    stale/expired; `reason` says which. Both mean "this node is up to date".
    """
    if body.op == "set":
        result = engine.apply_set(body.key, body.value, body.version, body.created_at, body.expires_at)
    else:
        result = engine.apply_delete(body.key, body.version)
    replication.record_received(result.reason, body.sent_at)
    return ApplyResponse(applied=result.applied, reason=result.reason, version=result.version)


@router.get("/replication/stats")
async def replication_stats(
    engine: CacheEngine = Depends(get_engine),
    replication: ReplicationManager = Depends(get_replication),
) -> dict:
    """Replication counters and recent failures. (Phase 8 folds these into /internal/metrics.)"""
    return {**replication.snapshot(), "tombstones": len(engine.versions)}
