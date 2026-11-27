"""Public API models for cache operations."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_KEY_LENGTH = 512


class CacheSetRequest(BaseModel):
    # extra="forbid": a misspelled field (e.g. "TTL") is a 422, not silently ignored.
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=MAX_KEY_LENGTH, examples=["user:101"])
    value: Any  # any JSON value, including null; required
    ttl: float | None = Field(default=None, gt=0, description="Seconds; omit for no expiry")

    @field_validator("key")
    @classmethod
    def key_has_no_slash(cls, key: str) -> str:
        # GET /cache/{key}/exists and /ttl use the key as a single path segment,
        # so a "/" in a key could not be read back reliably. Reject it up front.
        if "/" in key:
            raise ValueError("key must not contain '/'")
        return key


class CacheSetResponse(BaseModel):
    key: str
    created: bool
    version: int
    expires_at: float | None
    ttl_seconds: float | None
    node_id: str


class CacheGetResponse(BaseModel):
    key: str
    value: Any
    version: int
    expires_at: float | None
    ttl_seconds: float | None
    node_id: str


class CacheDeleteResponse(BaseModel):
    key: str
    deleted: bool
    node_id: str


class CacheExistsResponse(BaseModel):
    key: str
    exists: bool
    node_id: str


class CacheTTLResponse(BaseModel):
    key: str
    state: Literal["missing", "no_expiry", "expiring"]
    ttl_seconds: float | None
    node_id: str
