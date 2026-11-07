"""App 版本发布元数据（检查更新）。

单一发布记录表 app_release（id=1 当前行），运维通过 UPDATE 发布新版：
  UPDATE app_release SET version_code=4, version_name='1.3',
    download_url='https://.../app.apk', notes='更新说明';
GET /app/version 公开免认证（未登录也应能检查更新）。
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.database import SessionLocal
from sqlalchemy import text

router = APIRouter()


class AppVersionRead(BaseModel):
    version_code: int
    version_name: str
    download_url: str | None = None
    notes: str | None = None
    force_update: bool = False
    released_at: str | None = None


@router.get("/version", response_model=AppVersionRead)
async def get_latest_version() -> AppVersionRead:
    with SessionLocal() as db:
        row = db.execute(text(
            "select version_code, version_name, download_url, notes, "
            "force_update, released_at from app_release where id = 1"
        )).fetchone()
    if row is None:
        # 表未初始化：返回 0 版本占位（客户端按"无发布信息"处理）
        return AppVersionRead(version_code=0, version_name="0.0")
    return AppVersionRead(
        version_code=row[0],
        version_name=row[1],
        download_url=row[2],
        notes=row[3],
        force_update=bool(row[4]),
        released_at=str(row[5]) if row[5] else None,
    )
