"""Protects /internal/* endpoints with the shared cluster secret."""
from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from app.config import settings


def require_cluster_secret(x_cluster_secret: str | None = Header(default=None)) -> None:
    """FastAPI dependency: reject requests without the right X-Cluster-Secret.

    hmac.compare_digest compares in constant time, so response timing does not
    leak how many leading characters of a guessed secret were correct.
    """
    if x_cluster_secret is None or not hmac.compare_digest(
        x_cluster_secret.encode(), settings.cluster_secret.encode()
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing cluster secret",
        )