"""Engine-dispatched validation for immutable avatar packages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class AvatarPackageValidationError(ValueError):
    """The package is unsafe, corrupt, or violates its renderer contract."""


@dataclass(frozen=True)
class AvatarPackageValidationResult:
    engine: str
    renderer_mode: str
    package_format_version: int
    package_schema_revision: int
    manifest: dict[str, Any]
    report: dict[str, Any]


class AvatarPackageValidator(Protocol):
    def validate(self, package_bytes: bytes) -> AvatarPackageValidationResult: ...


class AvatarPackageValidatorRegistry:
    def __init__(self) -> None:
        self._validators: dict[str, AvatarPackageValidator] = {}

    def register(self, engine: str, validator: AvatarPackageValidator) -> None:
        normalized = engine.strip().lower()
        if not normalized:
            raise ValueError("engine must not be empty")
        self._validators[normalized] = validator

    def validate(
        self, engine: str, package_bytes: bytes
    ) -> AvatarPackageValidationResult:
        normalized = engine.strip().lower()
        validator = self._validators.get(normalized)
        if validator is None and normalized == "dhlive":
            # Lazy import keeps the generic registry free of engine-specific
            # dependencies while still making the built-in validator reliable.
            from app.services import dhlive_package_validator  # noqa: F401

            validator = self._validators.get(normalized)
        if validator is None:
            raise AvatarPackageValidationError(
                f"No package validator is registered for engine {engine!r}"
            )
        result = validator.validate(package_bytes)
        if result.engine != normalized:
            raise AvatarPackageValidationError(
                f"Validator returned engine {result.engine!r}, expected {engine!r}"
            )
        return result


avatar_package_validator_registry = AvatarPackageValidatorRegistry()
