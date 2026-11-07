"""Role-based access control for the admin API."""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, Request


@dataclass(frozen=True)
class AdminPrincipal:
    user_id: str | None
    role: str
    authenticated_by: str

    @property
    def audit_user_id(self) -> str | None:
        return self.user_id if self.authenticated_by == "jwt" else None


def require_admin(request: Request) -> AdminPrincipal:
    role = getattr(request.state, "role", None)
    if role not in {"admin", "super_admin"}:
        raise HTTPException(status_code=403, detail="admin access required")
    return AdminPrincipal(
        user_id=getattr(request.state, "user_id", None),
        role=role,
        authenticated_by=getattr(request.state, "authenticated_by", "unknown"),
    )


def require_super_admin(request: Request) -> AdminPrincipal:
    principal = require_admin(request)
    if principal.role != "super_admin" or principal.authenticated_by != "jwt":
        raise HTTPException(status_code=403, detail="super administrator account required")
    return principal


__all__ = ["AdminPrincipal", "require_admin", "require_super_admin"]
