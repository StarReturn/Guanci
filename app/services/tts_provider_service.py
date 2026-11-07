"""Database-backed TTS provider configuration."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import desc, select, update
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.core.database import SessionLocal, TtsProviderRecord
from app.core.secret_crypto import decrypt_secret, encrypt_secret, mask_secret

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TtsRuntimeConfig:
    type: str
    api_key: str | None
    workspace_id: str | None
    endpoint: str | None
    model: str | None
    default_voice_id: str | None
    source: str


class TtsProviderService:
    def list(self) -> list[TtsProviderRecord]:
        with SessionLocal() as db:
            return list(
                db.execute(
                    select(TtsProviderRecord).order_by(desc(TtsProviderRecord.created_at))
                ).scalars().all()
            )

    def get(self, provider_id: str) -> TtsProviderRecord | None:
        with SessionLocal() as db:
            return db.get(TtsProviderRecord, provider_id)

    def get_active(self) -> TtsProviderRecord | None:
        with SessionLocal() as db:
            return db.execute(
                select(TtsProviderRecord)
                .where(TtsProviderRecord.is_active.is_(True))
                .limit(1)
            ).scalar_one_or_none()

    def create(self, **values) -> TtsProviderRecord:
        if values.get("api_key"):
            values["api_key"] = encrypt_secret(values["api_key"])
        with SessionLocal() as db:
            rec = TtsProviderRecord(
                provider_id=str(uuid4()),
                **{key: value for key, value in values.items() if value is not None},
            )
            if rec.is_active:
                db.execute(update(TtsProviderRecord).values(is_active=False))
            db.add(rec)
            db.commit()
            db.refresh(rec)
            return rec

    def update(self, provider_id: str, **values) -> TtsProviderRecord | None:
        if values.get("api_key"):
            values["api_key"] = encrypt_secret(values["api_key"])
        activate = values.pop("is_active", None)
        with SessionLocal() as db:
            rec = db.get(TtsProviderRecord, provider_id)
            if rec is None:
                return None
            for key, value in values.items():
                if value is not None:
                    setattr(rec, key, value)
            if activate is True:
                db.execute(
                    update(TtsProviderRecord).values(is_active=False)
                )
                rec.is_active = True
            elif activate is False:
                rec.is_active = False
            db.commit()
            db.refresh(rec)
            return rec

    def delete(self, provider_id: str) -> bool:
        with SessionLocal() as db:
            rec = db.get(TtsProviderRecord, provider_id)
            if rec is None:
                return False
            db.delete(rec)
            db.commit()
            return True

    @staticmethod
    def masked_api_key(rec: TtsProviderRecord) -> str | None:
        return mask_secret(rec.api_key) if rec.api_key else None

    def runtime(self) -> TtsRuntimeConfig:
        try:
            active = self.get_active()
        except SQLAlchemyError as exc:
            logger.warning("tts_provider_db_unavailable: %s", exc)
            active = None
        if active is not None:
            return TtsRuntimeConfig(
                type=active.type.strip().lower(),
                api_key=decrypt_secret(active.api_key) if active.api_key else None,
                workspace_id=active.workspace_id,
                endpoint=active.endpoint,
                model=active.model,
                default_voice_id=active.default_voice_id,
                source="database",
            )
        return TtsRuntimeConfig(
            type=settings.tts_provider.strip().lower(),
            api_key=settings.dashscope_api_key.strip() or None,
            workspace_id=settings.aliyun_cosyvoice_workspace_id.strip() or None,
            endpoint=None,
            model=settings.aliyun_cosyvoice_model,
            default_voice_id=settings.aliyun_cosyvoice_default_voice_id.strip() or None,
            source="environment",
        )

    def runtime_for(self, provider_type: str) -> TtsRuntimeConfig:
        """Load the newest retained config for cleanup of historical resources."""
        with SessionLocal() as db:
            record = db.execute(
                select(TtsProviderRecord)
                .where(TtsProviderRecord.type == provider_type)
                .order_by(desc(TtsProviderRecord.updated_at))
                .limit(1)
            ).scalar_one_or_none()
        if record is not None:
            return TtsRuntimeConfig(
                type=record.type,
                api_key=decrypt_secret(record.api_key) if record.api_key else None,
                workspace_id=record.workspace_id,
                endpoint=record.endpoint,
                model=record.model,
                default_voice_id=record.default_voice_id,
                source="database",
            )
        runtime = self.runtime()
        if runtime.type != provider_type:
            raise RuntimeError(f"no retained TTS config for provider {provider_type}")
        return runtime


tts_provider_service = TtsProviderService()

__all__ = ["TtsRuntimeConfig", "TtsProviderService", "tts_provider_service"]
