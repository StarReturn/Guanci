"""Admin routes for user management.

Mounted on the admin app (port 8090). Requires ``X-API-Key: <ADMIN_API_KEY>``.
"""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response

from app.api.dependencies.admin import AdminPrincipal, require_admin

from app.schemas.auth import (
    AdminQuotaUpdate,
    AdminUserUpdate,
    AuditLogListResponse,
    AuditLogRead,
    UserListResponse,
    UserQuotaRead,
    UserRead,
    UserSessionListResponse,
    UserSessionRead,
)
from app.services.auth_service import (
    audit_service,
    quota_service,
    session_service,
    user_service,
)


router = APIRouter(dependencies=[Depends(require_admin)])


def _get_end_user(user_id: str):
    user = user_service.get_by_id(user_id)
    if user is None or user.role in {"admin", "super_admin"}:
        raise HTTPException(status_code=404, detail="user not found")
    return user


def _to_user_read(u) -> UserRead:
    return UserRead(
        user_id=u.user_id,
        phone=u.phone,
        email=u.email,
        wechat_openid=u.wechat_openid,
        nickname=u.nickname,
        avatar_url=u.avatar_url,
        status=u.status,
        role=u.role,
        phone_verified_at=u.phone_verified_at,
        last_login_at=u.last_login_at,
        created_at=u.created_at,
    )


@router.get("", response_model=UserListResponse)
async def list_users(
    status: str | None = Query(default=None),
    role: str | None = Query(default=None),
    search: str | None = Query(default=None, description="按手机号/昵称/邮箱模糊搜索"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> UserListResponse:
    if role in {"admin", "super_admin"}:
        raise HTTPException(status_code=400, detail="use the administrator management API")
    items, total = user_service.list(
        status_filter=status, role_filter=role, search=search,
        allowed_roles={"user", "vip"}, limit=limit, offset=offset,
    )
    return UserListResponse(items=[_to_user_read(u) for u in items], total=total)


@router.get("/{user_id}", response_model=UserRead)
async def get_user(user_id: str) -> UserRead:
    u = _get_end_user(user_id)
    return _to_user_read(u)


@router.patch("/{user_id}", response_model=UserRead)
async def update_user(
    user_id: str,
    body: AdminUserUpdate,
    request: Request,
    principal: AdminPrincipal = Depends(require_admin),
) -> UserRead:
    _get_end_user(user_id)
    u = user_service.admin_update(
        user_id, nickname=body.nickname, status=body.status, role=body.role,
    )
    if u is None:
        raise HTTPException(status_code=404, detail="user not found")
    audit_service.log(
        action="admin.user.update",
        actor_user_id=principal.audit_user_id,
        actor_ip=request.client.host if request.client else None,
        target_type="user", target_id=user_id,
        detail={"nickname": body.nickname, "status": body.status, "role": body.role},
    )
    return _to_user_read(u)


@router.delete("/{user_id}", response_model=UserRead)
async def disable_user(
    user_id: str,
    request: Request,
    principal: AdminPrincipal = Depends(require_admin),
) -> UserRead:
    """Soft delete: status=deleted. Audit log captures the disable."""
    _get_end_user(user_id)
    u = user_service.admin_update(user_id, status="deleted")
    if u is None:
        raise HTTPException(status_code=404, detail="user not found")
    # Also revoke all sessions
    n = session_service.revoke_all_for_user(user_id)
    audit_service.log(
        action="admin.user.disable",
        actor_user_id=principal.audit_user_id,
        actor_ip=request.client.host if request.client else None,
        target_type="user", target_id=user_id,
        detail={"revoked_sessions": n},
    )
    return _to_user_read(u)


# ----- Sessions: kick user offline -----

@router.get("/{user_id}/sessions", response_model=UserSessionListResponse)
async def list_user_sessions(user_id: str) -> UserSessionListResponse:
    _get_end_user(user_id)
    rows = session_service.list_for_user(user_id)
    return UserSessionListResponse(
        items=[
            UserSessionRead(
                jit=r.jit, device_info=r.device_info, ip=r.ip,
                expires_at=r.expires_at, revoked_at=r.revoked_at,
                created_at=r.created_at,
            )
            for r in rows
        ],
        total=len(rows),
    )


@router.delete("/{user_id}/sessions")
async def revoke_all_user_sessions(user_id: str) -> dict:
    """Force-logout everywhere. Use when you suspect account compromise."""
    _get_end_user(user_id)
    n = session_service.revoke_all_for_user(user_id)
    audit_service.log(
        action="admin.user.force_logout",
        target_type="user", target_id=user_id,
        detail={"revoked_count": n},
    )
    return {"revoked": n}


# ----- Quota (read-only stats; PATCH removed per decision 2026-06-23) -----

@router.get("/{user_id}/quota", response_model=UserQuotaRead)
async def get_user_quota(user_id: str) -> UserQuotaRead:
    """Usage stats only — no enforcement. Returns daily counters."""
    _get_end_user(user_id)
    return quota_service.get(user_id)


# ----- Audit log (global; per-user view is via target_id filter) -----

audit_router = APIRouter(dependencies=[Depends(require_admin)])


@audit_router.get("", response_model=AuditLogListResponse)
async def list_audit_logs(
    actor_user_id: str | None = Query(default=None),
    action: str | None = Query(default=None),
    target_type: str | None = Query(default=None),
    target_id: str | None = Query(default=None),
    from_at: datetime | None = Query(default=None),
    to_at: datetime | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    principal: AdminPrincipal = Depends(require_admin),
) -> AuditLogListResponse:
    if principal.role != "super_admin":
        if principal.user_id is None:
            raise HTTPException(status_code=403, detail="administrator account required")
        if actor_user_id and actor_user_id != principal.user_id:
            raise HTTPException(status_code=403, detail="administrators may only view their own audit log")
        actor_user_id = principal.user_id
    items, total = audit_service.list(
        actor_user_id=actor_user_id, action=action,
        target_type=target_type, target_id=target_id,
        from_at=from_at, to_at=to_at,
        limit=limit, offset=offset,
    )
    return AuditLogListResponse(
        items=[
            AuditLogRead(
                audit_id=r.audit_id,
                actor_user_id=r.actor_user_id,
                actor_admin_api_key_id=r.actor_admin_api_key_id,
                actor_ip=r.actor_ip,
                action=r.action,
                target_type=r.target_type,
                target_id=r.target_id,
                detail=r.detail,
                created_at=r.created_at,
            )
            for r in items
        ],
        total=total,
    )


def _csv_safe(value) -> str:
    text = "" if value is None else str(value)
    return "'" + text if text.startswith(("=", "+", "-", "@")) else text


@audit_router.get("/export")
async def export_audit_logs(
    actor_user_id: str | None = Query(default=None),
    action: str | None = Query(default=None),
    target_type: str | None = Query(default=None),
    target_id: str | None = Query(default=None),
    from_at: datetime | None = Query(default=None),
    to_at: datetime | None = Query(default=None),
    limit: int = Query(default=10000, ge=1, le=50000),
    principal: AdminPrincipal = Depends(require_admin),
) -> Response:
    if from_at and to_at and from_at >= to_at:
        raise HTTPException(status_code=422, detail="from_at must be earlier than to_at")
    if principal.role != "super_admin":
        if principal.user_id is None:
            raise HTTPException(status_code=403, detail="administrator account required")
        if actor_user_id and actor_user_id != principal.user_id:
            raise HTTPException(status_code=403, detail="administrators may only export their own audit log")
        actor_user_id = principal.user_id
    items, _ = audit_service.list(
        actor_user_id=actor_user_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        from_at=from_at,
        to_at=to_at,
        limit=limit,
        offset=0,
    )
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([
        "audit_id", "created_at", "actor_user_id", "actor_admin_api_key_id",
        "actor_ip", "action", "target_type", "target_id", "detail_json",
    ])
    for row in items:
        writer.writerow([
            _csv_safe(row.audit_id),
            row.created_at.isoformat(),
            _csv_safe(row.actor_user_id),
            _csv_safe(row.actor_admin_api_key_id),
            _csv_safe(row.actor_ip),
            _csv_safe(row.action),
            _csv_safe(row.target_type),
            _csv_safe(row.target_id),
            _csv_safe(json.dumps(row.detail or {}, ensure_ascii=False, separators=(",", ":"))),
        ])
    filename = f"audit-log-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    return Response(
        content="\ufeff" + output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
