"""Role-based LLM provider bindings for the memory subsystem.

Roles:

* ``chat`` — the conversational model. Migration seeds this row from the
  currently active ``llm_providers`` row, so existing chat behaviour is
  preserved without reconfiguration.
* ``memory_extract`` — the structured-JSON extractor. MUST be configured
  explicitly by a super admin. Never silently falls back to chat.
* ``memory_rerank`` — reserved for a future reranker; not yet invoked.

Even when chat and memory_extract point at the same physical provider, both
rows must exist. ``get_for_role('memory_extract')`` raises
``MemoryProviderNotConfigured`` if the binding is missing or disabled; the
caller then surfaces a structured error instead of degrading quality by
silently using chat.

``probe(provider_id, role)`` issues a structured JSON self-test so admins
cannot activate a binding that does not actually return JSON. The probe
payload is stored in ``probe_status`` / ``probe_message`` and inspected
before any worker invocation.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import desc, select, update

from app.core.database import (
    LlmProviderBindingRecord,
    LlmProviderRecord,
    SessionLocal,
)
from app.services.llm_service import llm_provider_service

logger = logging.getLogger(__name__)


class MemoryProviderError(RuntimeError):
    pass


class MemoryProviderNotConfigured(MemoryProviderError):
    """The requested role has no enabled binding."""


class MemoryProviderProbeFailed(MemoryProviderError):
    """The provider rejected a structured probe."""


@dataclass(frozen=True)
class BindingView:
    binding_id: str
    role: str
    provider_id: str
    enabled: bool
    probe_status: str
    probe_message: str | None
    config_overrides: dict


_ROLES = ("chat", "memory_extract", "memory_rerank")


class MemoryProviderService:
    # ------------------------------------------------------------------ read
    def list_bindings(self) -> list[BindingView]:
        with SessionLocal() as db:
            rows = list(db.execute(
                select(LlmProviderBindingRecord).order_by(
                    LlmProviderBindingRecord.role
                )
            ).scalars().all())
            return [self._view(row) for row in rows]

    def get_binding(self, role: str) -> BindingView | None:
        with SessionLocal() as db:
            row = db.execute(
                select(LlmProviderBindingRecord).where(
                    LlmProviderBindingRecord.role == role
                )
            ).scalar_one_or_none()
            return self._view(row) if row else None

    def _view(self, row: LlmProviderBindingRecord) -> BindingView:
        return BindingView(
            binding_id=row.binding_id, role=row.role,
            provider_id=row.provider_id, enabled=bool(row.enabled),
            probe_status=row.probe_status, probe_message=row.probe_message,
            config_overrides=dict(row.config_overrides or {}),
        )

    # ----------------------------------------------------------------- write
    def upsert_binding(
        self, *, role: str, provider_id: str, enabled: bool = True,
        config_overrides: dict | None = None,
    ) -> BindingView:
        if role not in _ROLES:
            raise MemoryProviderError(f"unknown provider role: {role}")
        with SessionLocal() as db:
            provider = db.get(LlmProviderRecord, provider_id)
            if provider is None:
                raise MemoryProviderError(
                    f"provider_id {provider_id} does not exist"
                )
            row = db.execute(
                select(LlmProviderBindingRecord).where(
                    LlmProviderBindingRecord.role == role
                )
            ).scalar_one_or_none()
            now = datetime.now(UTC)
            payload = dict(config_overrides or {})
            if row is None:
                row = LlmProviderBindingRecord(
                    binding_id=str(uuid4()), role=role,
                    provider_id=provider_id, enabled=enabled,
                    config_overrides=payload,
                    probe_status="pending", probed_at=None,
                )
                db.add(row)
            else:
                row.provider_id = provider_id
                row.enabled = enabled
                row.config_overrides = payload
                row.probe_status = "pending"
                row.probe_message = None
                row.probed_at = None
            db.commit()
            db.refresh(row)
            return self._view(row)

    def set_enabled(self, role: str, enabled: bool) -> BindingView:
        with SessionLocal() as db:
            row = db.execute(
                select(LlmProviderBindingRecord).where(
                    LlmProviderBindingRecord.role == role
                )
            ).scalar_one_or_none()
            if row is None:
                raise MemoryProviderError(f"binding for role {role} not configured")
            row.enabled = enabled
            db.commit()
            db.refresh(row)
            return self._view(row)

    def delete_binding(self, role: str) -> bool:
        with SessionLocal() as db:
            row = db.execute(
                select(LlmProviderBindingRecord).where(
                    LlmProviderBindingRecord.role == role
                )
            ).scalar_one_or_none()
            if row is None:
                return False
            db.delete(row)
            db.commit()
            return True

    # --------------------------------------------------------------- resolve
    def get_provider_for_role(self, role: str) -> LlmProviderRecord:
        """Return the provider for ``role`` or raise.

        Every role requires an enabled binding that passed its probe. Migration
        seeds ``chat`` from the legacy active provider, so normal upgrades stay
        non-breaking while later admin disable/rotation decisions take effect.
        """
        with SessionLocal() as db:
            row = db.execute(
                select(LlmProviderBindingRecord).where(
                    LlmProviderBindingRecord.role == role
                )
            ).scalar_one_or_none()
            if row is not None and row.enabled and row.probe_status == "ok":
                provider = db.get(LlmProviderRecord, row.provider_id)
                if provider is None:
                    raise MemoryProviderNotConfigured(
                        f"{role} binding references missing provider {row.provider_id}"
                    )
                return provider
            if row is None:
                raise MemoryProviderNotConfigured(
                    f"{role} provider binding is not configured; "
                    "super admin must bind and probe it explicitly"
                )
            if not row.enabled:
                raise MemoryProviderNotConfigured(f"{role} provider binding is disabled")
            raise MemoryProviderNotConfigured(
                f"{role} provider binding has not passed a structured probe; "
                f"probe_status={row.probe_status}"
            )

    # ----------------------------------------------------------------- probe
    async def probe(self, role: str) -> BindingView:
        """Issue a structured self-test and record the result.

        The probe is a tiny JSON-extraction prompt; the provider must reply
        with the exact JSON object. Any deviation marks the binding
        ``failed`` and the activator refuses to enable it.
        """
        binding = self.get_binding(role)
        if binding is None:
            raise MemoryProviderError(f"binding for role {role} not configured")
        with SessionLocal() as db:
            provider = db.get(LlmProviderRecord, binding.provider_id)
            if provider is None:
                raise MemoryProviderError(
                    f"binding {binding.binding_id} references missing provider"
                )
        try:
            from app.services.llm_service import llm_provider_service
            raw = await llm_provider_service.chat(
                [
                    {"role": "system", "content": "You are a strict JSON emitter."},
                    {"role": "user", "content": (
                        "Reply with exactly this JSON object and nothing else: "
                        "{\"ok\":true,\"role\":\"probe\"}"
                    )},
                ],
                provider=provider,
            )
        except Exception as exc:
            self._record_probe(role, ok=False, message=str(exc)[:500])
            raise MemoryProviderProbeFailed(
                f"{role} probe failed to call provider: {exc}"
            ) from exc
        try:
            parsed = json.loads(self._strip_codefence(raw))
        except json.JSONDecodeError as exc:
            self._record_probe(role, ok=False, message=f"non-JSON response: {raw[:200]}")
            raise MemoryProviderProbeFailed(
                f"{role} probe returned non-JSON response"
            ) from exc
        if not isinstance(parsed, dict) or parsed.get("ok") is not True:
            self._record_probe(role, ok=False, message=f"unexpected body: {raw[:200]}")
            raise MemoryProviderProbeFailed(
                f"{role} probe returned unexpected body: {parsed!r}"
            )
        self._record_probe(role, ok=True, message="probe ok")
        return self.get_binding(role)  # type: ignore[return-value]

    @staticmethod
    def _strip_codefence(raw: str) -> str:
        value = (raw or "").strip()
        if value.startswith("```"):
            import re
            value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I)
        return value

    def _record_probe(self, role: str, *, ok: bool, message: str) -> None:
        with SessionLocal() as db:
            db.execute(update(LlmProviderBindingRecord).where(
                LlmProviderBindingRecord.role == role
            ).values(
                probe_status="ok" if ok else "failed",
                probe_message=message[:500],
                probed_at=datetime.now(UTC),
            ))
            db.commit()


memory_provider_service = MemoryProviderService()


__all__ = [
    "BindingView",
    "MemoryProviderError",
    "MemoryProviderNotConfigured",
    "MemoryProviderProbeFailed",
    "MemoryProviderService",
    "memory_provider_service",
]
