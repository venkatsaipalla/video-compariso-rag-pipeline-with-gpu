import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from app.config import settings

API_KEY_HEADER = "X-API-Key"

_api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def require_api_key(provided: str | None = Depends(_api_key_header)) -> None:
    # Auth disabled when no key is configured (dev convenience).
    if not settings.RETRIEVAL_API_KEY:
        return
    if not provided or not secrets.compare_digest(provided, settings.RETRIEVAL_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing API key",
        )
