"""JWT encode/Decode utilities.

Token shape (HS256):

    Header:  {"alg": "HS256", "typ": "JWT"}
    Payload: {
        "sub": "<user_id>",          # subject (user UUID)
        "jit": "<jwt_id>",           # unique id; same as user_sessions.jit
        "iat": <issued_at>,
        "exp": <expires_at>,
        "iss": "<jwt_issuer>",
        "type": "access" | "refresh",
        "role": "<user_role>",       # "user" | "vip" | "admin"
    }

We hand-roll encode/decode rather than depend on PyJWT because PyJWT
isn't already in the dependency list and we don't need its advanced
features (JWE, JWK rotation). HS256 + a stable secret is fine for our
deployment model (single backend, single tenant, secret in env).

The encode/decode uses stdlib hmac + json + base64 — no extra deps.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from datetime import UTC, datetime, timedelta
from typing import Literal

from app.core.config import settings

logger = logging.getLogger(__name__)


_TOKEN_TYPE = Literal["access", "refresh"]


class JWTError(Exception):
    """Raised on signature mismatch, expired token, or malformed payload."""


def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _sign(message: bytes, secret: str) -> bytes:
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).digest()


def encode(
    *,
    user_id: str,
    role: str,
    token_type: _TOKEN_TYPE,
    jit: str | None = None,
    ttl_sec: int | None = None,
) -> tuple[str, dict]:
    """Sign and return ``(token_str, payload_dict)``.

    ``payload_dict`` is the decoded payload, returned so callers can
    persist ``jit``/``exp`` in user_sessions without re-decoding.
    """
    if not settings.jwt_secret:
        raise JWTError("jwt_secret is empty; cannot sign tokens")

    if jit is None:
        jit = secrets.token_hex(16)

    if ttl_sec is None:
        ttl_sec = (
            settings.jwt_access_token_ttl_sec
            if token_type == "access"
            else settings.jwt_refresh_token_ttl_sec
        )

    now = int(time.time())
    payload = {
        "sub": user_id,
        "jit": jit,
        "iat": now,
        "exp": now + ttl_sec,
        "iss": settings.jwt_issuer,
        "type": token_type,
        "role": role,
    }

    header = {"alg": settings.jwt_algorithm, "typ": "JWT"}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature_b64 = _b64url_encode(_sign(signing_input, settings.jwt_secret))

    token = f"{header_b64}.{payload_b64}.{signature_b64}"
    return token, payload


def decode(token: str) -> dict:
    """Verify signature + exp + iss. Raise JWTError on any failure."""
    if not settings.jwt_secret:
        raise JWTError("jwt_secret is empty")

    parts = token.split(".")
    if len(parts) != 3:
        raise JWTError("malformed token (expected 3 parts)")

    header_b64, payload_b64, signature_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected_sig = _sign(signing_input, settings.jwt_secret)

    try:
        provided_sig = _b64url_decode(signature_b64)
    except Exception as exc:
        raise JWTError(f"signature decode failed: {exc}") from exc

    if not hmac.compare_digest(expected_sig, provided_sig):
        raise JWTError("signature mismatch")

    try:
        header = json.loads(_b64url_decode(header_b64))
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception as exc:
        raise JWTError(f"payload decode failed: {exc}") from exc

    if header.get("alg") != settings.jwt_algorithm:
        raise JWTError(
            f"alg mismatch: expected {settings.jwt_algorithm}, got {header.get('alg')}"
        )

    now = int(time.time())
    if payload.get("exp", 0) < now:
        raise JWTError("token expired")

    if payload.get("iss") != settings.jwt_issuer:
        raise JWTError("iss mismatch")

    if payload.get("type") not in ("access", "refresh"):
        raise JWTError("invalid type claim")

    return payload


def issue_token_pair(
    *, user_id: str, role: str
) -> tuple[str, str, datetime, datetime]:
    """Convenience: issue access + refresh together.

    Returns ``(access_token, refresh_token, access_expires_at, refresh_expires_at)``.
    """
    a_token, a_payload = encode(
        user_id=user_id, role=role, token_type="access"
    )
    r_token, r_payload = encode(
        user_id=user_id, role=role, token_type="refresh"
    )
    a_exp = datetime.fromtimestamp(a_payload["exp"], tz=UTC)
    r_exp = datetime.fromtimestamp(r_payload["exp"], tz=UTC)
    return a_token, r_token, a_exp, r_exp


__all__ = ["encode", "decode", "issue_token_pair", "JWTError"]
