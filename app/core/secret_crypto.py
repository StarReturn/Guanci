"""Encryption helpers for provider credentials stored in the database."""
from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings

_PREFIX = "enc:v1:"


class SecretEncryptionError(RuntimeError):
    pass


def _fernet() -> Fernet:
    key = settings.provider_secret_encryption_key.strip()
    if not key:
        raise SecretEncryptionError(
            "PROVIDER_SECRET_ENCRYPTION_KEY is required for provider credential operations"
        )
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise SecretEncryptionError(
            "PROVIDER_SECRET_ENCRYPTION_KEY must be a valid Fernet key"
        ) from exc


def is_encrypted(value: str) -> bool:
    return value.startswith(_PREFIX)


def encrypt_secret(value: str) -> str:
    value = value.strip()
    if not value:
        raise SecretEncryptionError("provider credential cannot be empty")
    if is_encrypted(value):
        return value
    token = _fernet().encrypt(value.encode("utf-8")).decode("ascii")
    return f"{_PREFIX}{token}"


def decrypt_secret(value: str) -> str:
    """Decrypt v1 ciphertext; return legacy plaintext unchanged."""
    if not is_encrypted(value):
        return value
    try:
        return _fernet().decrypt(value[len(_PREFIX):].encode("ascii")).decode("utf-8")
    except (InvalidToken, UnicodeError) as exc:
        raise SecretEncryptionError(
            "provider credential cannot be decrypted with the configured key"
        ) from exc


def mask_secret(value: str) -> str:
    plaintext = decrypt_secret(value)
    if len(plaintext) <= 8:
        return "****"
    return f"{plaintext[:4]}****{plaintext[-4:]}"


__all__ = [
    "SecretEncryptionError",
    "decrypt_secret",
    "encrypt_secret",
    "is_encrypted",
    "mask_secret",
]
