"""Dedicated encryption helpers for sensitive long-term memory ciphertext.

This module intentionally keeps its own Fernet instance and key version tag
so that:

* rotating the memory key never touches JWT, provider, or storage secrets;
* if the provider-secret key is compromised, memory ciphertext remains safe;
* a future re-encrypt migration can migrate rows by ``encryption_key_version``
  without parsing provider ciphertext.

The ciphertext prefix ``mem:vN:`` carries the key version. ``encrypt()`` always
uses the configured ``settings.memory_encryption_key_version``. ``decrypt()``
inspects the prefix and refuses to silently fall back if the version is
unknown — operations fail closed instead of leaking plaintext.

Local dev allows an empty key (the helper raises a clear runtime error if a
sensitive memory is written without one). Production startup checks that the
key is present and differs from ``provider_secret_encryption_key``; see
``app.core.config.validate_production_security``.
"""
from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings


class MemoryEncryptionError(RuntimeError):
    pass


_PREFIX = "mem:"


def _fernet_for_version(version: int) -> Fernet:
    """Return the Fernet instance for the given key version.

    The current implementation keeps a single active key (version =
    ``settings.memory_encryption_key_version``). Future versions may add a
    keyring mapping for backward-compatible decryption during rotation.
    """
    if version != int(settings.memory_encryption_key_version):
        raise MemoryEncryptionError(
            f"memory encryption key version {version} is not registered"
        )
    key = (settings.memory_encryption_key or "").strip()
    if not key:
        raise MemoryEncryptionError(
            "MEMORY_ENCRYPTION_KEY is required to encrypt or decrypt sensitive memory"
        )
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise MemoryEncryptionError(
            "MEMORY_ENCRYPTION_KEY must be a valid Fernet key"
        ) from exc


def is_memory_ciphertext(value: str | None) -> bool:
    return bool(value) and value.startswith(_PREFIX)


def encrypt_memory(value: str) -> str:
    value = value.strip()
    if not value:
        raise MemoryEncryptionError("sensitive memory plaintext cannot be empty")
    version = int(settings.memory_encryption_key_version)
    token = _fernet_for_version(version).encrypt(value.encode("utf-8")).decode("ascii")
    return f"{_PREFIX}v{version}:{token}"


def decrypt_memory(value: str) -> str:
    if not is_memory_ciphertext(value):
        raise MemoryEncryptionError("value is not a memory ciphertext")
    body = value[len(_PREFIX):]
    if not body.startswith("v"):
        raise MemoryEncryptionError("memory ciphertext is missing version tag")
    try:
        version_str, token = body.split(":", 1)
    except ValueError as exc:
        raise MemoryEncryptionError("memory ciphertext format is invalid") from exc
    try:
        version = int(version_str[1:])
    except ValueError as exc:
        raise MemoryEncryptionError("memory ciphertext version is invalid") from exc
    try:
        return _fernet_for_version(version).decrypt(token.encode("ascii")).decode("utf-8")
    except (InvalidToken, UnicodeError) as exc:
        raise MemoryEncryptionError(
            "memory ciphertext cannot be decrypted with the configured key"
        ) from exc


def ensure_production_memory_key_is_independent() -> None:
    """Startup check: in production the memory key must be set and differ
    from the provider-secret key.

    Called by ``validate_production_security``; not exported in __all__ to
    discourage ad-hoc calls.
    """
    if settings.app_env.strip().lower() not in {"production", "prod"}:
        return
    memory_key = (settings.memory_encryption_key or "").strip()
    if not memory_key:
        raise RuntimeError(
            "MEMORY_ENCRYPTION_KEY is required in production for sensitive memory"
        )
    provider_key = (settings.provider_secret_encryption_key or "").strip()
    if memory_key == provider_key:
        raise RuntimeError(
            "MEMORY_ENCRYPTION_KEY must differ from PROVIDER_SECRET_ENCRYPTION_KEY"
        )


__all__ = [
    "MemoryEncryptionError",
    "decrypt_memory",
    "encrypt_memory",
    "is_memory_ciphertext",
]
