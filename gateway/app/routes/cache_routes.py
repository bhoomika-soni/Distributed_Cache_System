"""Public cache API. Handlers stay thin: validate, call the service, pick a status code."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Response, status

from app.deps import get_cache_service
from app.schemas.cache_schemas import (
    MAX_KEY_LENGTH,
    CacheDeleteResponse,
    CacheExistsResponse,
    CacheGetResponse,
    CacheSetRequest,
    CacheSetResponse,
    CacheTTLResponse,
)
from app.services.cache_service import CacheService

router = APIRouter(prefix="/cache", tags=["cache"])

# A key in the URL is a single path segment (no "/"), 1..512 characters.
KeyPath = Annotated[str, Path(min_length=1, max_length=MAX_KEY_LENGTH)]
Service = Annotated[CacheService, Depends(get_cache_service)]


@router.post("", response_model=CacheSetResponse, status_code=status.HTTP_201_CREATED)
async def set_key(body: CacheSetRequest, response: Response, service: Service):
    """201 when the key is new, 200 when an existing key was overwritten."""
    result = await service.set(body.key, body.value, body.ttl)
    if not result["created"]:
        response.status_code = status.HTTP_200_OK
    return result


@router.get("/{key}", response_model=CacheGetResponse)
async def get_key(key: KeyPath, service: Service):
    result = await service.get(key)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Key '{key}' not found")
    return result


@router.delete("/{key}", response_model=CacheDeleteResponse)
async def delete_key(key: KeyPath, service: Service):
    """Idempotent: deleting a missing key is 200 with deleted=false."""
    return await service.delete(key)


@router.get("/{key}/exists", response_model=CacheExistsResponse)
async def exists_key(key: KeyPath, service: Service):
    return await service.exists(key)


@router.get("/{key}/ttl", response_model=CacheTTLResponse)
async def ttl_key(key: KeyPath, service: Service):
    return await service.ttl(key)
