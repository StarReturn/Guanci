"""Pydantic schemas for client API key management.

Key design points:

* **Plaintext is shown only on create** — ``ClientApiKeyCreateResponse``
  is the ONLY schema that includes the plaintext ``key`` field. All
  read endpoints use ``ClientApiKeyRead`` which omits it.
* ``last_used_at`` and ``revoked_at`` are exposed for audit but are
  best-effort (``last_used_at`` is throttled to once per 60s).
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ClientApiKeyCreate(BaseModel):
    """Request body for POST /admin/api-keys."""

    description: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Human-readable label — e.g. 'iOS App v1.0 (prod)'",
    )
    # If omitted, server generates a fresh random key. Useful for tests
    # and for restoring from a known plaintext after DB wipe.
    plaintext_override: str | None = Field(
        default=None,
        min_length=16,
        max_length=128,
        description="Optional explicit plaintext (e.g. for restoring). "
                    "Default: auto-generate dh_client_<32 hex>.",
    )


class ClientApiKeyCreateResponse(BaseModel):
    """Response for POST /admin/api-keys — plaintext shown ONCE.

    Admin must copy the plaintext immediately; subsequent GETs will NOT
    return it. We log a warning server-side if the same key_id is
    fetched >5 times in 60s (suggests admin lost the plaintext and is
    trying to recover — impossible).
    """

    key_id: str
    key: str = Field(
        ...,
        description="Plaintext key — COPY NOW, will not be retrievable later",
    )
    description: str
    source: str
    created_at: datetime


class ClientApiKeyRead(BaseModel):
    """Read schema — no plaintext field."""

    key_id: str
    description: str
    source: str
    is_active: bool
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    # Masked hash prefix for debug/audit. First 8 chars only — enough
    # to confirm "yes this is the key I rotated" without leaking.
    key_hash_prefix: str = Field(
        ..., description="First 8 hex chars of sha256(salt+key), for audit"
    )


class ClientApiKeyUpdate(BaseModel):
    """Request body for PATCH /admin/api-keys/{id}."""

    description: str | None = Field(default=None, min_length=1, max_length=256)


class ClientApiKeyListResponse(BaseModel):
    """Paginated-ish response for GET /admin/api-keys."""

    items: list[ClientApiKeyRead]
    total: int
