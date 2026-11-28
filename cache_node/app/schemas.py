"""Request/response models for the node's internal API (validated by Pydantic)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReplicaTarget(BaseModel):
    """A node this write must be replicated to (chosen by the gateway's hash ring)."""

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1)
    url: str = Field(pattern=r"^https?://")


class SetRequest(BaseModel):
    # extra="forbid": a typo such as {"TTL": 60} is rejected with a 422 instead
    # of being silently ignored (which would store the key with no expiry).
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=512)
    value: Any  # any JSON value, including null. Required: the field has no default.
    ttl: float | None = Field(default=None, gt=0, description="Seconds; omit for no expiry")
    replicas: list[ReplicaTarget] = Field(default_factory=list, max_length=16)


class DeleteRequest(BaseModel):
    """Optional body of DELETE: tells the primary where to replicate the delete."""

    model_config = ConfigDict(extra="forbid")

    replicas: list[ReplicaTarget] = Field(default_factory=list, max_length=16)


class SetResponse(BaseModel):
    key: str
    created: bool  # True if the key is new, False if an existing key was overwritten
    version: int
    expires_at: float | None
    ttl_seconds: float | None
    replicas_scheduled: list[str]  # replication was STARTED for these nodes (async)


class GetResponse(BaseModel):
    key: str
    value: Any
    version: int
    created_at: float
    last_accessed_at: float
    expires_at: float | None
    ttl_seconds: float | None


class DeleteResponse(BaseModel):
    key: str
    deleted: bool
    replicas_scheduled: list[str]


class ExistsResponse(BaseModel):
    key: str
    exists: bool


class TTLResponse(BaseModel):
    key: str
    # missing    -> key absent or expired   (ttl_seconds is null)
    # no_expiry  -> key exists, lives forever (ttl_seconds is null)
    # expiring   -> key exists with a TTL     (ttl_seconds is the time left)
    state: Literal["missing", "no_expiry", "expiring"]
    ttl_seconds: float | None


# ------------------------------------------------------------------ replication
class ApplyRequest(BaseModel):
    """One replicated write, sent by a primary to a replica.

    Carries the primary's VERSION and ABSOLUTE expiry time (`expires_at`), never
    a relative TTL, so replicas expire the key at the same moment as the primary.
    """

    model_config = ConfigDict(extra="forbid")

    op: Literal["set", "delete"]
    key: str = Field(min_length=1, max_length=512)
    version: int = Field(ge=1)
    origin: str = Field(min_length=1)  # node id of the primary (for logs)
    sent_at: float  # primary's clock when replication was scheduled (lag measurement)
    # only for op == "set":
    value: Any = None
    created_at: float | None = None
    expires_at: float | None = None

    @model_validator(mode="after")
    def check_fields_for_op(self) -> "ApplyRequest":
        if self.op == "set":
            if "value" not in self.model_fields_set:  # null is a valid value; absent is not
                raise ValueError("'value' is required when op is 'set'")
            if self.created_at is None:
                raise ValueError("'created_at' is required when op is 'set'")
        else:
            extras = {"value", "created_at", "expires_at"} & self.model_fields_set
            if extras:
                raise ValueError(f"{sorted(extras)} are not allowed when op is 'delete'")
        return self


class ApplyResponse(BaseModel):
    applied: bool
    reason: Literal["applied", "stale", "expired"]
    version: int


class SnapshotEntry(BaseModel):
    key: str
    value: Any
    version: int
    created_at: float
    expires_at: float | None  # absolute expiry, so the receiver does not restart the TTL


class SnapshotResponse(BaseModel):
    node_id: str
    count: int
    entries: list[SnapshotEntry]
