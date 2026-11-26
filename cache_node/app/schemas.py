"""Request/response models for the node's internal API (validated by Pydantic)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SetRequest(BaseModel):
    # extra="forbid": a typo such as {"TTL": 60} is rejected with a 422 instead
    # of being silently ignored (which would store the key with no expiry).
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=512)
    value: Any  # any JSON value, including null. Required: the field has no default.
    ttl: float | None = Field(default=None, gt=0, description="Seconds; omit for no expiry")


class SetResponse(BaseModel):
    key: str
    created: bool  # True if the key is new, False if an existing key was overwritten
    version: int
    expires_at: float | None
    ttl_seconds: float | None


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
