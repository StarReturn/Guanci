"""Pydantic schemas for auth + user + audit.

Single file to keep import surface small; split into separate modules
later if this grows past ~500 lines.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ===========================================================================
# SMS code
# ===========================================================================

class SendCodeRequest(BaseModel):
    phone: str = Field(..., pattern=r"^\+?[0-9]{6,15}$", description="手机号，含或不含国际区号")
    purpose: Literal["register", "login", "reset_password"] = "login"


class SendCodeResponse(BaseModel):
    sent: bool
    resend_after_sec: int
    # Only populated when APP_ENV != production and sms_provider=mock,
    # so tests/dev can fetch the code without a real SMS gateway.
    dev_code: str | None = None


class SmsLoginRequest(BaseModel):
    phone: str = Field(..., pattern=r"^\+?[0-9]{6,15}$")
    code: str = Field(..., min_length=4, max_length=8, description="验证码")
    nickname: str | None = Field(default=None, max_length=64)


# ===========================================================================
# Password auth (Q10 decision: phone + password)
# ===========================================================================

class PasswordRegisterRequest(BaseModel):
    phone: str = Field(..., pattern=r"^\+?[0-9]{6,15}$")
    password: str = Field(..., min_length=6, max_length=128)
    nickname: str | None = Field(default=None, max_length=64)


class PasswordLoginRequest(BaseModel):
    phone: str = Field(..., pattern=r"^\+?[0-9]{6,15}$")
    password: str = Field(..., min_length=1, max_length=128)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=6, max_length=128)


class ForgotPasswordRequest(BaseModel):
    phone: str = Field(..., pattern=r"^\+?[0-9]{6,15}$")


class ResetPasswordRequest(BaseModel):
    phone: str = Field(..., pattern=r"^\+?[0-9]{6,15}$")
    code: str = Field(..., min_length=4, max_length=8, description="短信验证码")
    new_password: str = Field(..., min_length=6, max_length=128)


# ===========================================================================
# Admin login (Q: 管理端登录接口)
# ===========================================================================

class AdminLoginRequest(BaseModel):
    """Admin 登录 — username + password。"""
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class AdminCreateRequest(BaseModel):
    """Super administrator creates an administrator with an initial password."""
    username: str = Field(..., min_length=2, max_length=64)
    nickname: str = Field(default="管理员", max_length=64)
    password: str = Field(..., min_length=12, max_length=128)


class AdminRead(BaseModel):
    """管理员信息（列表/详情）。"""
    user_id: str
    username: str | None = None
    phone: str | None = None
    nickname: str
    avatar_url: str | None = None
    status: str
    role: str  # admin | super_admin
    last_login_at: str | None = None
    created_at: str


class AdminListResponse(BaseModel):
    items: list[AdminRead]
    total: int


class AdminPatchRequest(BaseModel):
    """超管修改管理员（改角色/封禁/改昵称）。"""
    nickname: str | None = Field(default=None, max_length=64)
    role: str | None = Field(default=None, description="admin | super_admin")
    status: str | None = Field(default=None, description="active | disabled")


# ===========================================================================
# WeChat OAuth
# ===========================================================================

class WechatLoginRequest(BaseModel):
    """Mobile app receives code from WeChat SDK, posts it here."""
    code: str = Field(..., min_length=1, max_length=256)
    nickname: str | None = Field(default=None, max_length=64)
    avatar_url: str | None = Field(default=None, max_length=512)


class WechatAuthUrlResponse(BaseModel):
    """For web-based flow. Mobile apps usually use SDK directly and skip this."""
    url: str


# ===========================================================================
# Token responses
# ===========================================================================

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["Bearer"] = "Bearer"
    expires_in: int = Field(..., description="access_token 寱秒 TTL")
    user: "UserRead"


class RefreshRequest(BaseModel):
    refresh_token: str


# ===========================================================================
# User
# ===========================================================================

class UserRead(BaseModel):
    user_id: str
    phone: str | None = None
    email: str | None = None
    wechat_openid: str | None = None
    nickname: str
    avatar_url: str | None = None
    status: str
    role: str
    phone_verified_at: datetime | None = None
    last_login_at: datetime | None = None
    created_at: datetime


class UserUpdate(BaseModel):
    nickname: str | None = Field(default=None, min_length=1, max_length=64)
    avatar_url: str | None = Field(default=None, max_length=512)


class AdminUserUpdate(BaseModel):
    """Admin-only user mutation (role/status)."""
    nickname: str | None = Field(default=None, max_length=64)
    status: Literal["active", "disabled", "deleted"] | None = None
    role: Literal["user", "vip"] | None = None


class UserListResponse(BaseModel):
    items: list[UserRead]
    total: int


# ===========================================================================
# Quota
# ===========================================================================

class UserQuotaRead(BaseModel):
    user_id: str
    quota_tts_daily: int
    quota_build_daily: int
    quota_download_daily: int
    used_tts_today: int
    used_build_today: int
    used_download_today: int
    reset_at: datetime


class AdminQuotaUpdate(BaseModel):
    quota_tts_daily: int | None = None
    quota_build_daily: int | None = None
    quota_download_daily: int | None = None


# ===========================================================================
# Avatar binding
# ===========================================================================

class UserAvatarRead(BaseModel):
    user_avatar_id: str
    avatar_id: str
    is_primary: bool
    custom_name: str | None = None
    created_at: datetime


class UserAvatarListResponse(BaseModel):
    items: list[UserAvatarRead]
    total: int


# ===========================================================================
# Audit
# ===========================================================================

class AuditLogRead(BaseModel):
    audit_id: str
    actor_user_id: str | None = None
    actor_admin_api_key_id: str | None = None
    actor_ip: str | None = None
    action: str
    target_type: str | None = None
    target_id: str | None = None
    detail: dict | None = None
    created_at: datetime


class AuditLogListResponse(BaseModel):
    items: list[AuditLogRead]
    total: int


# ===========================================================================
# Session (for admin's "kick user offline")
# ===========================================================================

class UserSessionRead(BaseModel):
    jit: str
    device_info: str | None = None
    ip: str | None = None
    expires_at: datetime
    revoked_at: datetime | None = None
    created_at: datetime


class UserSessionListResponse(BaseModel):
    items: list[UserSessionRead]
    total: int


# Resolve forward refs
TokenResponse.model_rebuild()
