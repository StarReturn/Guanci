"""Notification + Background services (站内信 + 背景图).

Lightweight — no APNs/FCM. Client polls GET /me/notifications.
"""
from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import desc, select, update, delete

from app.core.database import BackgroundRecord, NotificationRecord, SessionLocal
from app.services.storage import storage_service


class NotificationService:
    def send(self, *, user_id: str, type: str, title: str, body: str | None = None) -> NotificationRecord:
        with SessionLocal() as db:
            rec = NotificationRecord(
                notification_id=str(uuid4()),
                user_id=user_id, type=type, title=title, body=body,
            )
            db.add(rec)
            db.commit()
            db.refresh(rec)
        return rec

    def list(self, user_id: str, *, unread_only: bool = False) -> list[NotificationRecord]:
        with SessionLocal() as db:
            stmt = select(NotificationRecord).where(NotificationRecord.user_id == user_id)
            if unread_only:
                stmt = stmt.where(NotificationRecord.is_read.is_(False))
            stmt = stmt.order_by(desc(NotificationRecord.created_at)).limit(50)
            return list(db.execute(stmt).scalars().all())

    def mark_read(self, notification_id: str, user_id: str) -> bool:
        with SessionLocal() as db:
            result = db.execute(
                update(NotificationRecord)
                .where(
                    NotificationRecord.notification_id == notification_id,
                    NotificationRecord.user_id == user_id,
                )
                .values(is_read=True, read_at=datetime.now(UTC))
            )
            db.commit()
            return result.rowcount > 0


    def delete(self, notification_id: str, user_id: str) -> bool:
        with SessionLocal() as db:
            result = db.execute(
                delete(NotificationRecord).where(
                    NotificationRecord.notification_id == notification_id,
                    NotificationRecord.user_id == user_id,
                )
            )
            db.commit()
            return result.rowcount > 0

    def clear(self, user_id: str, *, read_only: bool = True) -> int:
        """删除本用户通知：read_only=True 只清已读；返回删除条数。"""
        with SessionLocal() as db:
            stmt = delete(NotificationRecord).where(
                NotificationRecord.user_id == user_id)
            if read_only:
                stmt = stmt.where(NotificationRecord.is_read.is_(True))
            result = db.execute(stmt)
            db.commit()
            return result.rowcount

notification_service = NotificationService()


class BackgroundService:
    def list(self) -> list[BackgroundRecord]:
        with SessionLocal() as db:
            return list(db.execute(
                select(BackgroundRecord).order_by(desc(BackgroundRecord.created_at))
            ).scalars().all())

    def list_for_user(self, user_id: str) -> list[BackgroundRecord]:
        with SessionLocal() as db:
            return list(db.execute(
                select(BackgroundRecord)
                .where(
                    (BackgroundRecord.source == "preset")
                    | (BackgroundRecord.uploaded_by == user_id)
                )
                .order_by(desc(BackgroundRecord.created_at))
            ).scalars().all())

    def get(self, background_id: str) -> BackgroundRecord | None:
        with SessionLocal() as db:
            return db.get(BackgroundRecord, background_id)

    def create(self, *, name: str, content: bytes, content_type: str,
               source: str = "preset", uploaded_by: str | None = None) -> BackgroundRecord:
        bg_id = str(uuid4())
        ext = "jpg" if "jpeg" in content_type else "png" if "png" in content_type else "webp"
        key = f"backgrounds/{bg_id}.{ext}"
        storage_service.put_bytes(key=key, content=content, content_type=content_type)
        with SessionLocal() as db:
            rec = BackgroundRecord(
                background_id=bg_id, name=name, source=source,
                storage_key=key, uploaded_by=uploaded_by,
            )
            db.add(rec)
            db.commit()
            db.refresh(rec)
        return rec

    def delete(self, background_id: str) -> bool:
        with SessionLocal() as db:
            rec = db.get(BackgroundRecord, background_id)
            if rec is None:
                return False
            db.delete(rec)
            db.commit()
        return True

    def get_bytes(self, background_id: str) -> bytes | None:
        rec = self.get(background_id)
        if rec is None:
            return None
        return storage_service.get_bytes(rec.storage_key)


background_service = BackgroundService()


__all__ = ["notification_service", "background_service"]
