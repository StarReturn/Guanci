"""Client-side auth + me routes.

All under ``/api/v1/auth`` and ``/api/v1/me``.

Mounted on the client app (port 8091). Most endpoints are PUBLIC
(no JWT required) by design — you can't have a JWT before login.
``/me/*`` requires JWT.
"""
from __future__ import annotations

import os
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request, File, UploadFile

from app.core.config import settings
from app.middleware.jwt import get_request_user_id
from app.schemas.auth import (
    ChangePasswordRequest,
    ForgotPasswordRequest,
    PasswordLoginRequest,
    PasswordRegisterRequest,
    RefreshRequest,
    ResetPasswordRequest,
    SendCodeRequest,
    SendCodeResponse,
    SmsLoginRequest,
    TokenResponse,
    UserAvatarListResponse,
    UserAvatarRead,
    UserQuotaRead,
    UserRead,
    UserUpdate,
    WechatAuthUrlResponse,
    WechatLoginRequest,
)
from app.services.auth_service import (
    audit_service,
    auth_service,
    quota_service,
    session_service,
    sms_code_service,
    sms_rate_limit_service,
    user_avatar_service,
    user_service,
)
from app.services.notification_service import notification_service, background_service
from uuid import uuid4 as _uuid4

from app.services.storage import storage_service
from app.services.photo_validator import validate_photo, PhotoValidationError
from app.services.sms_service import sms_service
from app.services.wechat_service import wechat_service


router = APIRouter()


def _client_ip(request: Request) -> str | None:
    fwd = request.headers.get("X-Forwarded-For", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else None


def _is_dev_env() -> bool:
    return (settings.app_env or "local").lower() in ("local", "dev", "test")


def _avatar_public_url(avatar_url: str | None) -> str | None:
    """头像列存 MinIO key（user_avatar/ 前缀）时签发预签名 URL；外部完整 URL 透传。"""
    if avatar_url and "://" not in avatar_url:
        try:
            return storage_service.presigned_url(avatar_url)
        except Exception:  # noqa: BLE001
            return avatar_url
    return avatar_url


def _to_user_read(user) -> UserRead:
    return UserRead(
        user_id=user.user_id,
        phone=user.phone,
        email=user.email,
        wechat_openid=user.wechat_openid,
        nickname=user.nickname,
        avatar_url=_avatar_public_url(user.avatar_url),
        status=user.status,
        role=user.role,
        phone_verified_at=user.phone_verified_at,
        last_login_at=user.last_login_at,
        created_at=user.created_at,
    )


# ===========================================================================
# SMS code endpoints
# ===========================================================================

@router.post("/sms/send-code", response_model=SendCodeResponse)
async def send_sms_code(body: SendCodeRequest, request: Request) -> SendCodeResponse:
    ip = _client_ip(request)

    ok, reason = sms_rate_limit_service.check_can_send(phone=body.phone, ip=ip)
    if not ok:
        raise HTTPException(status_code=429, detail=reason)

    code_id, plaintext = sms_code_service.create(
        phone=body.phone, purpose=body.purpose, ip=ip
    )

    sent = sms_service.send(phone=body.phone, code=plaintext)

    # Dev convenience: return code in response so tests work without SMS
    dev_code = plaintext if (_is_dev_env() and settings.sms_provider == "mock") else None

    audit_service.log(
        action="sms.send_code",
        actor_ip=ip,
        target_type="phone", target_id=body.phone,
        detail={"sent": sent, "purpose": body.purpose},
    )

    return SendCodeResponse(
        sent=sent,
        resend_after_sec=settings.sms_resend_cooldown_sec,
        dev_code=dev_code,
    )


@router.post("/sms/login", response_model=TokenResponse)
async def sms_login(body: SmsLoginRequest, request: Request) -> TokenResponse:
    ip = _client_ip(request)
    result = auth_service.login_or_register_by_phone(body=body, ip=ip)
    if result is None:
        raise HTTPException(status_code=401, detail="invalid code or account disabled")
    user, access, refresh = result
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=settings.jwt_access_token_ttl_sec,
        user=_to_user_read(user),
    )


# ===========================================================================
# Password auth (Q10 decision: phone + password)
# ===========================================================================

def _issue_tokens_for_user(user, ip: str | None) -> tuple[str, str]:
    """Helper: issue JWT pair + persist refresh session."""
    from app.core.jwt_utils import decode as jwt_decode, issue_token_pair
    access, refresh, _, _ = issue_token_pair(user_id=user.user_id, role=user.role)
    r_payload = jwt_decode(refresh)
    session_service.create(
        user_id=user.user_id, refresh_token=refresh, ip=ip,
        device_info=None,
        expires_at=datetime.fromtimestamp(r_payload["exp"], tz=UTC),
        jit=r_payload["jit"],
    )
    return access, refresh


@router.post("/register", response_model=TokenResponse)
async def password_register(body: PasswordRegisterRequest, request: Request) -> TokenResponse:
    """Register with phone + password. Returns tokens immediately (auto-login)."""
    ip = _client_ip(request)
    try:
        user = user_service.register_with_password(
            phone=body.phone, password=body.password, nickname=body.nickname,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    user_service.mark_login(user.user_id, ip)
    audit_service.log(
        action="user.register", actor_user_id=user.user_id, actor_ip=ip,
        detail={"method": "password"},
    )
    access, refresh = _issue_tokens_for_user(user, ip)
    return TokenResponse(
        access_token=access, refresh_token=refresh,
        expires_in=settings.jwt_access_token_ttl_sec, user=_to_user_read(user),
    )


@router.post("/login", response_model=TokenResponse)
async def password_login(body: PasswordLoginRequest, request: Request) -> TokenResponse:
    """Login with phone + password."""
    ip = _client_ip(request)
    user = user_service.login_with_password(phone=body.phone, password=body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid phone or password")
    if user.status != "active":
        raise HTTPException(status_code=403, detail="account disabled")
    user_service.mark_login(user.user_id, ip)
    audit_service.log(
        action="user.login", actor_user_id=user.user_id, actor_ip=ip,
        detail={"method": "password"},
    )
    access, refresh = _issue_tokens_for_user(user, ip)
    return TokenResponse(
        access_token=access, refresh_token=refresh,
        expires_in=settings.jwt_access_token_ttl_sec, user=_to_user_read(user),
    )


@router.post("/change-password")
async def change_password(body: ChangePasswordRequest, request: Request) -> dict:
    """Change password — requires current password + JWT."""
    uid = get_request_user_id(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    ok = user_service.change_password(
        user_id=uid, current=body.current_password, new=body.new_password,
    )
    if not ok:
        raise HTTPException(status_code=400, detail="current password incorrect")
    # Revoke all sessions (force re-login with new password)
    session_service.revoke_all_for_user(uid)
    audit_service.log(
        action="user.change_password", actor_user_id=uid,
        actor_ip=_client_ip(request),
    )
    return {"ok": True, "detail": "password changed; please re-login"}


@router.post("/forgot-password")
async def forgot_password(body: ForgotPasswordRequest, request: Request) -> dict:
    """Trigger: send SMS code (purpose=reset_password).

    We don't reveal whether the phone exists or not — always return 200
    to prevent phone-enumeration attacks.
    """
    ip = _client_ip(request)
    ok, reason = sms_rate_limit_service.check_can_send(phone=body.phone, ip=ip)
    if not ok:
        raise HTTPException(status_code=429, detail=reason)
    code_id, plaintext = sms_code_service.create(
        phone=body.phone, purpose="reset_password", ip=ip,
    )
    sms_service.send(phone=body.phone, code=plaintext)
    dev_code = plaintext if (_is_dev_env() and settings.sms_provider == "mock") else None
    return {
        "sent": True,
        "resend_after_sec": settings.sms_resend_cooldown_sec,
        "dev_code": dev_code,
    }


@router.post("/reset-password")
async def reset_password(body: ResetPasswordRequest, request: Request) -> dict:
    """Reset password using SMS code from /forgot-password."""
    ip = _client_ip(request)
    if not sms_code_service.verify(
        phone=body.phone, code=body.code, purpose="reset_password",
    ):
        raise HTTPException(status_code=401, detail="invalid or expired code")
    user = user_service.reset_password(phone=body.phone, new=body.new_password)
    if user is None:
        raise HTTPException(status_code=404, detail="phone not registered")
    # Revoke all existing sessions (force re-login everywhere)
    session_service.revoke_all_for_user(user.user_id)
    audit_service.log(
        action="user.reset_password", actor_user_id=user.user_id,
        actor_ip=ip,
    )
    return {"ok": True, "detail": "password reset; please login with new password"}


# ===========================================================================
# WeChat OAuth
# ===========================================================================

@router.get("/wechat/url", response_model=WechatAuthUrlResponse)
async def wechat_auth_url(redirect_uri: str, state: str = "") -> WechatAuthUrlResponse:
    """For web flow. Mobile apps usually use the WeChat SDK and skip this."""
    return WechatAuthUrlResponse(
        url=wechat_service.get_auth_url(redirect_uri=redirect_uri, state=state)
    )


@router.post("/wechat/login", response_model=TokenResponse)
async def wechat_login(body: WechatLoginRequest, request: Request) -> TokenResponse:
    """Mobile posts the code from WeChat SDK here."""
    ip = _client_ip(request)

    # Async call to wechat_service
    info = await wechat_service.exchange_code_for_userinfo_async(body.code)
    if info is None or not info.get("openid"):
        raise HTTPException(status_code=401, detail="wechat code exchange failed")

    # Reuse auth_service flow but adapt: auth_service.login_by_wechat
    # expects sync. We adapt by storing info and calling a thin wrapper.
    result = await _do_wechat_login(body, info, ip)
    if result is None:
        raise HTTPException(status_code=401, detail="wechat login failed")
    user, access, refresh = result
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=settings.jwt_access_token_ttl_sec,
        user=_to_user_read(user),
    )


async def _do_wechat_login(body, info, ip):
    """Bridge to sync auth_service.login_by_wechat.

    auth_service.login_by_wechat already calls sync version; we
    rewrite here to avoid the sync/async split.
    """
    from app.core.jwt_utils import decode as jwt_decode
    from datetime import datetime, timezone

    user = user_service.get_by_wechat_openid(info["openid"])
    if user is None:
        user = user_service.create_from_wechat(
            openid=info["openid"],
            unionid=info.get("unionid"),
            nickname=body.nickname or info.get("nickname") or "微信用户",
            avatar_url=body.avatar_url or info.get("headimgurl"),
        )
        audit_service.log(
            action="user.register",
            actor_user_id=user.user_id, actor_ip=ip,
            detail={"method": "wechat", "openid": info["openid"]},
        )
    else:
        if user.status != "active":
            return None
        audit_service.log(
            action="user.login",
            actor_user_id=user.user_id, actor_ip=ip,
            detail={"method": "wechat"},
        )

    user_service.mark_login(user.user_id, ip)

    from app.core.jwt_utils import issue_token_pair
    from app.services.auth_service import session_service
    access, refresh, _, _ = issue_token_pair(user_id=user.user_id, role=user.role)
    r_payload = jwt_decode(refresh)
    session_service.create(
        user_id=user.user_id, refresh_token=refresh, ip=ip,
        device_info=None,
        expires_at=datetime.fromtimestamp(r_payload["exp"], tz=UTC),
        jit=r_payload["jit"],
    )
    return user, access, refresh


# ===========================================================================
# Token refresh + logout
# ===========================================================================

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(body: RefreshRequest) -> TokenResponse:
    result = auth_service.refresh_tokens(body.refresh_token)
    if result is None:
        raise HTTPException(status_code=401, detail="invalid or revoked refresh token")
    user, access, new_refresh = result
    return TokenResponse(
        access_token=access,
        refresh_token=new_refresh,
        expires_in=settings.jwt_access_token_ttl_sec,
        user=_to_user_read(user),
    )


@router.post("/logout")
async def logout(body: RefreshRequest) -> dict:
    ok = auth_service.logout(body.refresh_token)
    return {"logged_out": ok}


# ===========================================================================
# /me — current user info (requires JWT)
# ===========================================================================

me_router = APIRouter()


@me_router.get("", response_model=UserRead)
async def get_me(request: Request) -> UserRead:
    uid = get_request_user_id(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    user = user_service.get_by_id(uid)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    return _to_user_read(user)


@me_router.patch("", response_model=UserRead)
async def update_me(body: UserUpdate, request: Request) -> UserRead:
    uid = get_request_user_id(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    user = user_service.update_profile(
        uid, nickname=body.nickname, avatar_url=body.avatar_url
    )
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    return _to_user_read(user)


@me_router.post("/avatar")
async def upload_my_avatar(request: Request, file: UploadFile = File(...)) -> dict:
    """用户头像上传：multipart 直传，服务端转存 MinIO，返回预签名 URL。"""
    uid = get_request_user_id(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    content_type = file.content_type or "image/jpeg"
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="unsupported avatar image type")
    content = await file.read()
    if not content or len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="avatar must be 1 byte to 5 MB")
    ext = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}[content_type]
    key = f"user_avatar/{uid}/{_uuid4().hex}.{ext}"
    storage_service.put_bytes(key, content, content_type)
    user = user_service.update_profile(uid, nickname=None, avatar_url=key)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    return {"url": storage_service.presigned_url(key)}


@me_router.get("/quota", response_model=UserQuotaRead)
async def get_my_quota(request: Request) -> UserQuotaRead:
    uid = get_request_user_id(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    return quota_service.get(uid)


@me_router.get("/avatars", response_model=UserAvatarListResponse)
async def list_my_avatars(request: Request) -> UserAvatarListResponse:
    uid = get_request_user_id(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    items = user_avatar_service.list_for_user(uid)
    return UserAvatarListResponse(
        items=[
            UserAvatarRead(
                user_avatar_id=r.user_avatar_id,
                avatar_id=r.avatar_id,
                is_primary=r.is_primary,
                custom_name=r.custom_name,
                created_at=r.created_at,
            )
            for r in items
        ],
        total=len(items),
    )


@me_router.post("/avatars/{user_avatar_id}/primary")
async def set_primary_avatar(user_avatar_id: str, request: Request) -> dict:
    uid = get_request_user_id(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    ok = user_avatar_service.set_primary(uid, user_avatar_id)
    if not ok:
        raise HTTPException(status_code=404, detail="avatar binding not found")
    return {"ok": True}


# ===========================================================================
# Avatar binding — bind a keyframe package (built by admin) to the user
# ===========================================================================

from pydantic import BaseModel as _PydBase, Field as _PydField


class BindAvatarRequest(_PydBase):
    """Bind an existing avatar (admin must have built the package first)."""
    avatar_id: str = _PydField(..., min_length=1, max_length=160)
    custom_name: str | None = _PydField(default=None, max_length=64)


@me_router.post("/avatars", response_model=UserAvatarRead, status_code=201)
async def bind_avatar(body: BindAvatarRequest, request: Request) -> UserAvatarRead:
    uid = get_request_user_id(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="not authenticated")

    # Verify the avatar package actually exists (otherwise users could
    # bind random IDs and break rendering later).
    from app.services.digital_human_service import digital_human_service
    from app.services.keyframe_repository import keyframe_repository
    package = keyframe_repository.get_package(body.avatar_id)
    if package is None:
        raise HTTPException(
            status_code=404, detail="avatar package not found; ask admin to build it first"
        )
    managed = digital_human_service.get_by_id(package.digital_human_id)
    if managed is not None and not digital_human_service.user_owns_avatar(
        uid, body.avatar_id
    ):
        raise HTTPException(status_code=404, detail="avatar package not found")

    record = user_avatar_service.bind(
        user_id=uid, avatar_id=body.avatar_id, custom_name=body.custom_name,
    )
    audit_service.log(
        action="user.bind_avatar",
        actor_user_id=uid,
        target_type="avatar", target_id=body.avatar_id,
        detail={"user_avatar_id": record.user_avatar_id},
    )
    return UserAvatarRead(
        user_avatar_id=record.user_avatar_id,
        avatar_id=record.avatar_id,
        is_primary=record.is_primary,
        custom_name=record.custom_name,
        created_at=record.created_at,
    )


# ===========================================================================
# Self-service build — user uploads photo + selects background
# ===========================================================================

from app.services.keyframe_repository import keyframe_repository


class BuildStatusResponse(__import__("pydantic").BaseModel):
    task_id: str
    status: str  # queued | running | completed | failed
    avatar_id: str | None = None
    estimated_seconds: int = 300
    error: str | None = None
    stage: str | None = None       # queued|preparing|photo_activation|dhlive_build|validating|reference_render|activating|done
    progress: int = 0              # 0..100


@me_router.post("/avatars/build", status_code=202)
async def self_build_avatar(
    request: Request,
) -> BuildStatusResponse:
    """Retired direct-build endpoint kept only for an explicit migration error."""
    uid = get_request_user_id(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="not authenticated")

    raise HTTPException(
        status_code=422,
        detail=(
            "legacy direct avatar build is disabled for v3; call "
            "/api/v1/me/avatars/photo-preflight, then use the digital-human "
            "build endpoint with preprocess_token"
        ),
    )


@me_router.get("/avatars/build/{task_id}", response_model=BuildStatusResponse)
async def get_build_status(task_id: str, request: Request) -> BuildStatusResponse:
    """查构建进度（轮询用）。"""
    uid = get_request_user_id(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    task = keyframe_repository.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    if (task.owner_user_id or task.digital_human_id) != uid:
        raise HTTPException(status_code=404, detail="task not found")
    return BuildStatusResponse(
        task_id=task.task_id,
        status=task.status,
        avatar_id=task.avatar_id,
        error=task.error,
        stage=task.stage,
        progress=task.progress or 0,
    )


# ===========================================================================
# Notifications (站内信)
# ===========================================================================

from pydantic import BaseModel as _BM, Field as _FL


class NotificationRead(_BM):
    notification_id: str
    type: str
    title: str
    body: str | None = None
    is_read: bool
    created_at: str  # ISO format


class NotificationListResponse(_BM):
    items: list[NotificationRead]
    total: int
    unread_count: int


@me_router.get("/notifications", response_model=NotificationListResponse)
async def list_notifications(
    request: Request,
    unread_only: bool = False,
) -> NotificationListResponse:
    uid = get_request_user_id(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    items = notification_service.list(uid, unread_only=unread_only)
    all_items = notification_service.list(uid)
    unread = sum(1 for n in all_items if not n.is_read)
    return NotificationListResponse(
        items=[
            NotificationRead(
                notification_id=n.notification_id,
                type=n.type, title=n.title, body=n.body,
                is_read=n.is_read,
                created_at=n.created_at.isoformat(),
            )
            for n in items
        ],
        total=len(all_items),
        unread_count=unread,
    )


@me_router.post("/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: str, request: Request) -> dict:
    uid = get_request_user_id(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    ok = notification_service.mark_read(notification_id, uid)
    if not ok:
        raise HTTPException(status_code=404, detail="notification not found")
    return {"ok": True}


@me_router.delete("/notifications/{notification_id}")
async def delete_notification(notification_id: str, request: Request) -> dict:
    """删除单条通知（仅本人）。"""
    uid = get_request_user_id(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    ok = notification_service.delete(notification_id, uid)
    if not ok:
        raise HTTPException(status_code=404, detail="notification not found")
    return {"ok": True, "deleted": 1}


@me_router.delete("/notifications")
async def clear_notifications(request: Request, all: bool = False) -> dict:
    """清空通知。默认只清已读；?all=true 清全部（含未读，前端需二次确认）。"""
    uid = get_request_user_id(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    deleted = notification_service.clear(uid, read_only=not all)
    return {"ok": True, "deleted": deleted}


# ===========================================================================
# Backgrounds (user-visible list of preset + own backgrounds)
# ===========================================================================

class BackgroundRead(_BM):
    background_id: str
    name: str
    source: str
    preview_url: str | None = None


@me_router.get("/backgrounds", response_model=list[BackgroundRead])
async def list_backgrounds_for_user(request: Request) -> list[BackgroundRead]:
    """List all backgrounds available to this user (preset + own uploads)."""
    user_id = get_request_user_id(request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    bgs = background_service.list_for_user(user_id)
    return [
        BackgroundRead(
            background_id=b.background_id,
            name=b.name, source=b.source,
            preview_url=storage_service.presigned_url(b.storage_key) if hasattr(storage_service, 'presigned_url') else None,
        )
        for b in bgs
    ]


from uuid import uuid4 as _uuid4

from app.services.storage import storage_service


# ===========================================================================
# Dev-only test endpoints (APP_ENV=local/dev/test only)
# ===========================================================================

dev_router = APIRouter()


@dev_router.post("/login", response_model=TokenResponse)
async def dev_login(nickname: str = "dev", role: str = "user") -> TokenResponse:
    """Quick login without SMS — for tests/dev only.

    Creates (or reuses) a user with phone=dev-<random>-<nickname> and
    returns valid tokens. Disabled in production by app startup check.
    """
    if not _is_dev_env():
        raise HTTPException(status_code=404, detail="dev endpoints disabled in production")

    # Find or create a dev user
    dev_phone = f"+1000000{abs(hash(nickname)) % 100000:05d}"
    user = user_service.get_by_phone(dev_phone)
    if user is None:
        user = user_service.create_from_phone(phone=dev_phone, nickname=nickname)
        if role != "user":
            user_service.admin_update(user.user_id, role=role)

    from app.core.jwt_utils import decode as jwt_decode, issue_token_pair
    from app.services.auth_service import session_service
    access, refresh, _, _ = issue_token_pair(user_id=user.user_id, role=user.role)
    r_payload = jwt_decode(refresh)
    session_service.create(
        user_id=user.user_id, refresh_token=refresh, ip="dev",
        device_info="dev-login",
        expires_at=datetime.fromtimestamp(r_payload["exp"], tz=UTC),
        jit=r_payload["jit"],
    )
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=settings.jwt_access_token_ttl_sec,
        user=_to_user_read(user),
    )


# Production does not expose the route at all, including in OpenAPI.
if _is_dev_env():
    router.include_router(dev_router, prefix="/dev")
