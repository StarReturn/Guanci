"""Auth service — user CRUD + login + JWT issuance + audit + quota.

Single module to keep auth-related logic in one place. Split later if
this grows past ~500 lines.
"""
from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import UTC, datetime, timedelta
from typing import Iterable
from uuid import uuid4

from sqlalchemy import desc, func, select, update

from app.core.config import settings
from app.core.database import (
    AuditLogRecord,
    SessionLocal,
    SmsCodeRecord,
    UserAvatarRecord,
    UserQuotaRecord,
    UserRecord,
    UserSessionRecord,
)
from app.core.jwt_utils import JWTError, decode, issue_token_pair
from app.schemas.auth import (
    AdminQuotaUpdate,
    SmsLoginRequest,
    UserAvatarRead,
    UserQuotaRead,
    UserRead,
    UserSessionRead,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Hashing helpers
# ===========================================================================

def _hash_code(phone: str, code: str, salt: str) -> str:
    return hashlib.sha256(f"{phone}|{code}|{salt}".encode()).hexdigest()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _next_midnight_utc() -> datetime:
    now = datetime.now(UTC)
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


# ===========================================================================
# Users
# ===========================================================================

class UserService:
    """CRUD + login flows for end-users."""

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_by_id(self, user_id: str) -> UserRecord | None:
        with SessionLocal() as db:
            return db.get(UserRecord, user_id)

    def get_by_phone(self, phone: str) -> UserRecord | None:
        with SessionLocal() as db:
            return db.execute(
                select(UserRecord).where(UserRecord.phone == phone)
            ).scalar_one_or_none()

    def get_by_wechat_openid(self, openid: str) -> UserRecord | None:
        with SessionLocal() as db:
            return db.execute(
                select(UserRecord).where(UserRecord.wechat_openid == openid)
            ).scalar_one_or_none()

    def list(
        self,
        *,
        status_filter: str | None = None,
        role_filter: str | None = None,
        allowed_roles: set[str] | None = None,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[UserRecord], int]:
        with SessionLocal() as db:
            stmt = select(UserRecord)
            if status_filter:
                stmt = stmt.where(UserRecord.status == status_filter)
            if role_filter:
                stmt = stmt.where(UserRecord.role == role_filter)
            elif allowed_roles:
                stmt = stmt.where(UserRecord.role.in_(allowed_roles))
            if search:
                like = f"%{search}%"
                stmt = stmt.where(
                    (UserRecord.phone.like(like))
                    | (UserRecord.nickname.like(like))
                    | (UserRecord.email.like(like))
                )
            stmt = stmt.order_by(desc(UserRecord.created_at)).limit(limit).offset(offset)
            items = list(db.execute(stmt).scalars().all())
            # total (separate count for simplicity)
            count_stmt = select(UserRecord)
            if status_filter:
                count_stmt = count_stmt.where(UserRecord.status == status_filter)
            if role_filter:
                count_stmt = count_stmt.where(UserRecord.role == role_filter)
            elif allowed_roles:
                count_stmt = count_stmt.where(UserRecord.role.in_(allowed_roles))
            if search:
                count_stmt = count_stmt.where(
                    (UserRecord.phone.like(like))
                    | (UserRecord.nickname.like(like))
                    | (UserRecord.email.like(like))
                )
            total = len(list(db.execute(count_stmt).scalars().all()))
            return items, total

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def create_from_phone(self, *, phone: str, nickname: str | None = None) -> UserRecord:
        user = UserRecord(
            user_id=str(uuid4()),
            phone=phone,
            nickname=nickname or "用户",
            status="active",
            role="user",
            phone_verified_at=datetime.now(UTC),
        )
        with SessionLocal() as db:
            db.add(user)
            # Seed default quota
            db.add(UserQuotaRecord(
                user_id=user.user_id,
                quota_tts_daily=settings.default_quota_tts_daily,
                quota_build_daily=settings.default_quota_build_daily,
                quota_download_daily=settings.default_quota_download_daily,
                reset_at=_next_midnight_utc(),
            ))
            db.commit()
            db.refresh(user)
        return user

    def create_from_wechat(
        self,
        *,
        openid: str,
        unionid: str | None = None,
        nickname: str | None = None,
        avatar_url: str | None = None,
    ) -> UserRecord:
        user = UserRecord(
            user_id=str(uuid4()),
            wechat_openid=openid,
            wechat_unionid=unionid,
            nickname=nickname or "微信用户",
            avatar_url=avatar_url,
            status="active",
            role="user",
        )
        with SessionLocal() as db:
            db.add(user)
            db.add(UserQuotaRecord(
                user_id=user.user_id,
                quota_tts_daily=settings.default_quota_tts_daily,
                quota_build_daily=settings.default_quota_build_daily,
                quota_download_daily=settings.default_quota_download_daily,
                reset_at=_next_midnight_utc(),
            ))
            db.commit()
            db.refresh(user)
        return user

    def update_profile(
        self, user_id: str, *, nickname: str | None = None, avatar_url: str | None = None
    ) -> UserRecord | None:
        with SessionLocal() as db:
            user = db.get(UserRecord, user_id)
            if user is None:
                return None
            if nickname is not None:
                user.nickname = nickname
            if avatar_url is not None:
                user.avatar_url = avatar_url
            db.commit()
            db.refresh(user)
        return user

    def admin_update(
        self,
        user_id: str,
        *,
        nickname: str | None = None,
        status: str | None = None,
        role: str | None = None,
    ) -> UserRecord | None:
        with SessionLocal() as db:
            user = db.get(UserRecord, user_id)
            if user is None:
                return None
            if nickname is not None:
                user.nickname = nickname
            if status is not None:
                user.status = status
            if role is not None:
                user.role = role
            db.commit()
            db.refresh(user)
        return user

    def mark_login(self, user_id: str, ip: str | None) -> None:
        with SessionLocal() as db:
            db.execute(
                update(UserRecord)
                .where(UserRecord.user_id == user_id)
                .values(last_login_at=datetime.now(UTC), last_login_ip=ip)
            )
            db.commit()

    # ------------------------------------------------------------------
    # Password auth (Q10)
    # ------------------------------------------------------------------

    def register_with_password(self, *, phone: str, password: str, nickname: str | None) -> UserRecord:
        """Create a user with phone + bcrypt-hashed password."""
        import bcrypt
        if self.get_by_phone(phone) is not None:
            raise ValueError("phone already registered")
        pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")
        user = UserRecord(
            user_id=str(uuid4()),
            phone=phone,
            password_hash=pw_hash,
            nickname=nickname or "用户",
            status="active",
            role="user",
            phone_verified_at=datetime.now(UTC),
        )
        with SessionLocal() as db:
            db.add(user)
            db.add(UserQuotaRecord(
                user_id=user.user_id,
                quota_tts_daily=settings.default_quota_tts_daily,
                quota_build_daily=settings.default_quota_build_daily,
                quota_download_daily=settings.default_quota_download_daily,
                reset_at=_next_midnight_utc(),
            ))
            db.commit()
            db.refresh(user)
        return user

    def login_with_password(self, *, phone: str, password: str) -> UserRecord | None:
        """Verify password; return user if OK, None if wrong."""
        import bcrypt
        user = self.get_by_phone(phone)
        if user is None or user.password_hash is None:
            return None
        try:
            ok = bcrypt.checkpw(password.encode("utf-8"), user.password_hash.encode("utf-8"))
        except Exception:
            return None
        return user if ok else None

    def change_password(self, *, user_id: str, current: str, new: str) -> bool:
        """Returns True if changed, False if current password wrong."""
        import bcrypt
        with SessionLocal() as db:
            user = db.get(UserRecord, user_id)
            if user is None or user.password_hash is None:
                return False
            try:
                ok = bcrypt.checkpw(current.encode("utf-8"), user.password_hash.encode("utf-8"))
            except Exception:
                return False
            if not ok:
                return False
            user.password_hash = bcrypt.hashpw(new.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")
            db.commit()
        return True

    def reset_password(self, *, phone: str, new: str) -> UserRecord | None:
        """Used after SMS code verification in /forgot-password flow."""
        import bcrypt
        with SessionLocal() as db:
            user = db.execute(
                select(UserRecord).where(UserRecord.phone == phone)
            ).scalar_one_or_none()
            if user is None:
                return None
            user.password_hash = bcrypt.hashpw(new.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")
            db.commit()
            db.refresh(user)
        return user


user_service = UserService()


# ===========================================================================
# SMS codes
# ===========================================================================

class SmsCodeService:
    """Insert + verify SMS codes with per-row salt."""

    def create(
        self, *, phone: str, purpose: str, ip: str | None
    ) -> tuple[str, str]:
        """Generate and persist a 6-digit code. Returns (code_id, plaintext_code).

        Stores ``salt`` alongside ``code_hash`` so verify() can recompute
        with the correct salt (defeats rainbow tables even on DB dump).
        """
        code = f"{secrets.randbelow(1000000):06d}"
        salt = secrets.token_hex(8)
        code_hash = _hash_code(phone, code, salt)

        with SessionLocal() as db:
            record = SmsCodeRecord(
                code_id=str(uuid4()),
                phone=phone,
                code_hash=code_hash,
                salt=salt,
                purpose=purpose,
                expires_at=datetime.now(UTC) + timedelta(seconds=settings.sms_code_ttl_sec),
                ip=ip,
            )
            db.add(record)
            db.commit()
            db.refresh(record)
        return record.code_id, code

    def verify(self, *, phone: str, code: str, purpose: str) -> bool:
        """Mark code consumed if it matches an unconsumed unexpired record.

        Iterates over recent records and recomputes hash with each
        record's own salt. Constant-time compare via ``hmac.compare_digest``.
        """
        import hmac as _hmac
        now = datetime.now(UTC)
        with SessionLocal() as db:
            stmt = (
                select(SmsCodeRecord)
                .where(
                    SmsCodeRecord.phone == phone,
                    SmsCodeRecord.purpose == purpose,
                    SmsCodeRecord.consumed_at.is_(None),
                    SmsCodeRecord.expires_at > now,
                    SmsCodeRecord.created_at > now - timedelta(hours=24),
                )
                .order_by(desc(SmsCodeRecord.created_at))
            )
            recent = list(db.execute(stmt).scalars().all())
            for r in recent:
                expected = _hash_code(phone, code, r.salt)
                if _hmac.compare_digest(expected, r.code_hash):
                    r.consumed_at = now
                    db.commit()
                    return True
            return False


sms_code_service = SmsCodeService()


# ===========================================================================
# SMS rate limiting (per phone + per IP)
# ===========================================================================

class SmsRateLimitService:
    def check_can_send(self, *, phone: str, ip: str | None) -> tuple[bool, str]:
        """Return (ok, reason)."""
        now = datetime.now(UTC)
        cooldown_at = now - timedelta(seconds=settings.sms_resend_cooldown_sec)
        day_ago = now - timedelta(hours=24)

        with SessionLocal() as db:
            # Cooldown per phone
            recent_phone = db.execute(
                select(SmsCodeRecord)
                .where(SmsCodeRecord.phone == phone, SmsCodeRecord.created_at > cooldown_at)
                .limit(1)
            ).scalar_one_or_none()
            if recent_phone is not None:
                wait = int(
                    settings.sms_resend_cooldown_sec
                    - (now - recent_phone.created_at).total_seconds()
                )
                return False, f"resend cooldown ({max(wait, 1)}s)"

            if ip:
                ip_count = len(list(db.execute(
                    select(SmsCodeRecord)
                    .where(SmsCodeRecord.ip == ip, SmsCodeRecord.created_at > day_ago)
                ).scalars().all()))
                if ip_count >= settings.sms_ip_daily_limit:
                    return False, "ip daily limit exceeded"

            phone_count = len(list(db.execute(
                select(SmsCodeRecord)
                .where(SmsCodeRecord.phone == phone, SmsCodeRecord.created_at > day_ago)
            ).scalars().all()))
            if phone_count >= settings.sms_phone_daily_limit:
                return False, "phone daily limit exceeded"

        return True, "ok"


sms_rate_limit_service = SmsRateLimitService()


# ===========================================================================
# Sessions (refresh tokens)
# ===========================================================================

class SessionService:
    def create(
        self, *,
        user_id: str,
        refresh_token: str,
        ip: str | None,
        device_info: str | None,
        expires_at: datetime,
        jit: str,
    ) -> None:
        with SessionLocal() as db:
            db.add(UserSessionRecord(
                jit=jit,
                user_id=user_id,
                token_hash=_hash_token(refresh_token),
                ip=ip,
                device_info=device_info,
                expires_at=expires_at,
            ))
            db.commit()

    def revoke(self, jit: str) -> bool:
        with SessionLocal() as db:
            row = db.get(UserSessionRecord, jit)
            if row is None or row.revoked_at is not None:
                return False
            row.revoked_at = datetime.now(UTC)
            db.commit()
        return True

    def revoke_all_for_user(self, user_id: str) -> int:
        now = datetime.now(UTC)
        with SessionLocal() as db:
            rows = db.execute(
                select(UserSessionRecord).where(
                    UserSessionRecord.user_id == user_id,
                    UserSessionRecord.revoked_at.is_(None),
                )
            ).scalars().all()
            for r in rows:
                r.revoked_at = now
            db.commit()
            return len(rows)

    def verify_refresh(self, refresh_token: str) -> dict | None:
        """Decode + check session row exists and isn't revoked. Returns payload."""
        try:
            payload = decode(refresh_token)
        except JWTError:
            return None
        if payload.get("type") != "refresh":
            return None
        with SessionLocal() as db:
            row = db.get(UserSessionRecord, payload["jit"])
            if row is None or row.revoked_at is not None:
                return None
            if row.token_hash != _hash_token(refresh_token):
                return None
        return payload

    def list_for_user(self, user_id: str) -> list[UserSessionRecord]:
        with SessionLocal() as db:
            return list(db.execute(
                select(UserSessionRecord)
                .where(UserSessionRecord.user_id == user_id)
                .order_by(desc(UserSessionRecord.created_at))
            ).scalars().all())


session_service = SessionService()


# ===========================================================================
# Quota
# ===========================================================================

class QuotaService:
    """Lazy daily-reset + check + consume."""

    def _get_or_create(self, db, user_id: str) -> UserQuotaRecord:
        rec = db.get(UserQuotaRecord, user_id)
        if rec is None:
            rec = UserQuotaRecord(
                user_id=user_id,
                quota_tts_daily=settings.default_quota_tts_daily,
                quota_build_daily=settings.default_quota_build_daily,
                quota_download_daily=settings.default_quota_download_daily,
                reset_at=_next_midnight_utc(),
            )
            db.add(rec)
            db.flush()
        # Lazy reset
        if rec.reset_at < datetime.now(UTC):
            rec.used_tts_today = 0
            rec.used_build_today = 0
            rec.used_download_today = 0
            rec.reset_at = _next_midnight_utc()
        return rec

    def get(self, user_id: str) -> UserQuotaRead:
        with SessionLocal() as db:
            rec = self._get_or_create(db, user_id)
            db.commit()
            return UserQuotaRead(
                user_id=rec.user_id,
                quota_tts_daily=rec.quota_tts_daily,
                quota_build_daily=rec.quota_build_daily,
                quota_download_daily=rec.quota_download_daily,
                used_tts_today=rec.used_tts_today,
                used_build_today=rec.used_build_today,
                used_download_today=rec.used_download_today,
                reset_at=rec.reset_at,
            )

    def check(self, user_id: str, kind: str) -> bool:
        with SessionLocal() as db:
            rec = self._get_or_create(db, user_id)
            quota = getattr(rec, f"quota_{kind}_daily")
            used = getattr(rec, f"used_{kind}_today")
            return used < quota

    def consume(self, user_id: str, kind: str) -> None:
        with SessionLocal() as db:
            rec = self._get_or_create(db, user_id)
            current = getattr(rec, f"used_{kind}_today")
            setattr(rec, f"used_{kind}_today", current + 1)
            db.commit()

    def admin_update(
        self, user_id: str, body: AdminQuotaUpdate
    ) -> UserQuotaRead | None:
        with SessionLocal() as db:
            rec = self._get_or_create(db, user_id)
            if body.quota_tts_daily is not None:
                rec.quota_tts_daily = body.quota_tts_daily
            if body.quota_build_daily is not None:
                rec.quota_build_daily = body.quota_build_daily
            if body.quota_download_daily is not None:
                rec.quota_download_daily = body.quota_download_daily
            db.commit()
            db.refresh(rec)
        return self.get(user_id)


quota_service = QuotaService()


# ===========================================================================
# Avatars (user → 数字人 binding)
# ===========================================================================

class UserAvatarService:
    def list_for_user(self, user_id: str) -> list[UserAvatarRecord]:
        with SessionLocal() as db:
            return list(db.execute(
                select(UserAvatarRecord)
                .where(UserAvatarRecord.user_id == user_id)
                .order_by(desc(UserAvatarRecord.created_at))
            ).scalars().all())

    def owns_avatar(self, user_id: str, avatar_id: str) -> bool:
        with SessionLocal() as db:
            return db.execute(
                select(UserAvatarRecord.user_avatar_id).where(
                    UserAvatarRecord.user_id == user_id,
                    UserAvatarRecord.avatar_id == avatar_id,
                ).limit(1)
            ).scalar_one_or_none() is not None

    def bind(
        self,
        *,
        user_id: str,
        avatar_id: str,
        digital_human_id: str | None = None,
        custom_name: str | None = None,
        voice_ref_audio_key: str | None = None,
        voice_ref_text: str | None = None,
        voice_ref_duration_ms: int | None = None,
        voice_provider: str | None = None,
        voice_model_id: str | None = None,
        voice_provider_voice_id: str | None = None,
    ) -> UserAvatarRecord:
        """Bind an avatar to a user.

        Voice-cloning fields (P3 batch 4, inline mode): pass them in when
        the avatar was built from a user-uploaded voice sample. They stay
        NULL for admin-built packages or when no voice sample was uploaded.
        """
        with SessionLocal() as db:
            if digital_human_id:
                current = db.execute(
                    select(UserAvatarRecord).where(
                        UserAvatarRecord.user_id == user_id,
                        UserAvatarRecord.digital_human_id == digital_human_id,
                    )
                ).scalar_one_or_none()
                if current is not None:
                    current.avatar_id = avatar_id
                    current.custom_name = custom_name
                    current.voice_ref_audio_key = voice_ref_audio_key
                    current.voice_ref_text = voice_ref_text
                    current.voice_ref_duration_ms = voice_ref_duration_ms
                    current.voice_provider = voice_provider
                    current.voice_model_id = voice_model_id
                    current.voice_provider_voice_id = voice_provider_voice_id
                    db.commit()
                    db.refresh(current)
                    return current
            # First binding becomes primary
            existing = db.execute(
                select(UserAvatarRecord).where(UserAvatarRecord.user_id == user_id)
            ).scalars().all()
            is_primary = len(existing) == 0
            record = UserAvatarRecord(
                user_avatar_id=str(uuid4()),
                user_id=user_id,
                digital_human_id=digital_human_id,
                avatar_id=avatar_id,
                is_primary=is_primary,
                custom_name=custom_name,
                voice_ref_audio_key=voice_ref_audio_key,
                voice_ref_text=voice_ref_text,
                voice_ref_duration_ms=voice_ref_duration_ms,
                voice_provider=voice_provider,
                voice_model_id=voice_model_id,
                voice_provider_voice_id=voice_provider_voice_id,
            )
            db.add(record)
            db.commit()
            db.refresh(record)
        return record

    def get_primary_for_user(self, user_id: str) -> UserAvatarRecord | None:
        """Return the user's primary avatar binding, or None if no binding yet.

        Used by /dialogue (and the WebSocket realtime endpoint) to look
        up the ``voice_model_id`` for TTS ``reference_id``.
        """
        with SessionLocal() as db:
            return db.execute(
                select(UserAvatarRecord).where(
                    UserAvatarRecord.user_id == user_id,
                    UserAvatarRecord.is_primary.is_(True),
                )
            ).scalar_one_or_none()

    def set_primary(self, user_id: str, user_avatar_id: str) -> bool:
        with SessionLocal() as db:
            target = db.execute(
                select(UserAvatarRecord).where(
                    UserAvatarRecord.user_avatar_id == user_avatar_id,
                    UserAvatarRecord.user_id == user_id,
                )
            ).scalar_one_or_none()
            if target is None:
                return False
            # Unset siblings
            siblings = db.execute(
                select(UserAvatarRecord).where(UserAvatarRecord.user_id == user_id)
            ).scalars().all()
            for s in siblings:
                s.is_primary = (s.user_avatar_id == user_avatar_id)
            db.commit()
        return True

    def set_provider_voice(
        self,
        *,
        user_avatar_id: str,
        provider: str,
        model_id: str,
        provider_voice_id: str,
    ) -> bool:
        with SessionLocal() as db:
            record = db.get(UserAvatarRecord, user_avatar_id)
            if record is None:
                return False
            record.voice_provider = provider
            record.voice_model_id = model_id
            record.voice_provider_voice_id = provider_voice_id
            db.commit()
        return True


user_avatar_service = UserAvatarService()


# ===========================================================================
# Audit log
# ===========================================================================

class AuditService:
    def log(
        self,
        *,
        action: str,
        actor_user_id: str | None = None,
        actor_admin_api_key_id: str | None = None,
        actor_ip: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        detail: dict | None = None,
    ) -> None:
        """Fire-and-forget. Never raises — audit must not break request flow."""
        try:
            with SessionLocal() as db:
                db.add(AuditLogRecord(
                    audit_id=str(uuid4()),
                    actor_user_id=actor_user_id,
                    actor_admin_api_key_id=actor_admin_api_key_id,
                    actor_ip=actor_ip,
                    action=action,
                    target_type=target_type,
                    target_id=target_id,
                    detail=detail,
                ))
                db.commit()
        except Exception as exc:
            logger.warning("audit_log_write_failed", extra={"error": str(exc)})

    def list(
        self, *,
        actor_user_id: str | None = None,
        action: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        from_at: datetime | None = None,
        to_at: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[AuditLogRecord], int]:
        with SessionLocal() as db:
            stmt = select(AuditLogRecord)
            count_stmt = select(func.count()).select_from(AuditLogRecord)
            filters = []
            if actor_user_id:
                filters.append(AuditLogRecord.actor_user_id == actor_user_id)
            if action:
                filters.append(AuditLogRecord.action == action)
            if target_type:
                filters.append(AuditLogRecord.target_type == target_type)
            if target_id:
                filters.append(AuditLogRecord.target_id == target_id)
            if from_at:
                filters.append(AuditLogRecord.created_at >= from_at)
            if to_at:
                filters.append(AuditLogRecord.created_at < to_at)
            if filters:
                stmt = stmt.where(*filters)
                count_stmt = count_stmt.where(*filters)
            stmt = stmt.order_by(desc(AuditLogRecord.created_at)).limit(limit).offset(offset)
            items = list(db.execute(stmt).scalars().all())
            return items, int(db.scalar(count_stmt) or 0)


audit_service = AuditService()


# ===========================================================================
# Top-level auth flow (login / register / refresh)
# ===========================================================================

class AuthService:
    """Top-level: combines user + sms + session + audit."""

    def login_or_register_by_phone(
        self, *, body: SmsLoginRequest, ip: str | None
    ) -> tuple[UserRecord, str, str] | None:
        """Verify code; create user if first time; issue tokens.

        Returns (user, access, refresh) or None on bad code.
        """
        if not sms_code_service.verify(
            phone=body.phone, code=body.code, purpose="login"
        ):
            return None

        user = user_service.get_by_phone(body.phone)
        if user is None:
            user = user_service.create_from_phone(
                phone=body.phone, nickname=body.nickname
            )
            audit_service.log(
                action="user.register",
                actor_user_id=user.user_id,
                actor_ip=ip,
                target_type="user", target_id=user.user_id,
                detail={"method": "sms", "phone": body.phone},
            )
        else:
            if user.status != "active":
                return None  # disabled/deleted accounts can't log in
            audit_service.log(
                action="user.login",
                actor_user_id=user.user_id,
                actor_ip=ip,
                target_type="user", target_id=user.user_id,
                detail={"method": "sms"},
            )

        user_service.mark_login(user.user_id, ip)

        access, refresh, _, refresh_exp = issue_token_pair(
            user_id=user.user_id, role=user.role
        )
        # We need jit + exp from refresh token — decode it back
        # (cheap, signature already verified by encode).
        from app.core.jwt_utils import decode as jwt_decode
        r_payload = jwt_decode(refresh)
        session_service.create(
            user_id=user.user_id,
            refresh_token=refresh,
            ip=ip,
            device_info=None,  # TODO: parse UA
            expires_at=datetime.fromtimestamp(r_payload["exp"], tz=UTC),
            jit=r_payload["jit"],
        )
        return user, access, refresh

    def login_by_wechat(
        self, *, code: str, nickname: str | None, avatar_url: str | None, ip: str | None
    ) -> tuple[UserRecord, str, str] | None:
        """Exchange WeChat code for openid; create/login user."""
        from app.services.wechat_service import wechat_service
        info = wechat_service.exchange_code_for_userinfo(code)
        if info is None or not info.get("openid"):
            return None

        user = user_service.get_by_wechat_openid(info["openid"])
        if user is None:
            user = user_service.create_from_wechat(
                openid=info["openid"],
                unionid=info.get("unionid"),
                nickname=nickname or info.get("nickname"),
                avatar_url=avatar_url or info.get("headimgurl"),
            )
            audit_service.log(
                action="user.register",
                actor_user_id=user.user_id,
                actor_ip=ip,
                detail={"method": "wechat", "openid": info["openid"]},
            )
        else:
            if user.status != "active":
                return None
            audit_service.log(
                action="user.login",
                actor_user_id=user.user_id,
                actor_ip=ip,
                detail={"method": "wechat"},
            )

        user_service.mark_login(user.user_id, ip)

        access, refresh, _, _ = issue_token_pair(
            user_id=user.user_id, role=user.role
        )
        from app.core.jwt_utils import decode as jwt_decode
        r_payload = jwt_decode(refresh)
        session_service.create(
            user_id=user.user_id,
            refresh_token=refresh,
            ip=ip,
            device_info=None,
            expires_at=datetime.fromtimestamp(r_payload["exp"], tz=UTC),
            jit=r_payload["jit"],
        )
        return user, access, refresh

    def refresh_tokens(self, refresh_token: str) -> tuple[UserRecord, str, str] | None:
        payload = session_service.verify_refresh(refresh_token)
        if payload is None:
            return None
        user = user_service.get_by_id(payload["sub"])
        if user is None or user.status != "active":
            return None

        # Rotate: revoke old, issue new
        session_service.revoke(payload["jit"])

        access, new_refresh, _, _ = issue_token_pair(
            user_id=user.user_id, role=user.role
        )
        from app.core.jwt_utils import decode as jwt_decode
        r_payload = jwt_decode(new_refresh)
        session_service.create(
            user_id=user.user_id,
            refresh_token=new_refresh,
            expires_at=datetime.fromtimestamp(r_payload["exp"], tz=UTC),
            jit=r_payload["jit"],
            ip=None,
            device_info=None,
        )
        return user, access, new_refresh

    def logout(self, refresh_token: str) -> bool:
        """Revoke session for given refresh token."""
        try:
            payload = decode(refresh_token)
        except JWTError:
            return False
        return session_service.revoke(payload.get("jit", ""))


auth_service = AuthService()


__all__ = [
    "user_service", "sms_code_service", "sms_rate_limit_service",
    "session_service", "quota_service", "user_avatar_service",
    "audit_service", "auth_service",
]
