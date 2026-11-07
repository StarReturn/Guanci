"""预设音色库 API：列表（含新鲜试听 URL）。"""
from __future__ import annotations

from urllib.parse import urlparse

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from app.core.database import SessionLocal
from sqlalchemy import text

router = APIRouter()


class PresetVoiceRead(BaseModel):
    voice_id: str
    display_name: str
    gender: str
    age_band: str
    style: str | None = None
    accent: str | None = None
    sample_url: str | None = None


@router.get("", response_model=list[PresetVoiceRead])
async def list_preset_voices(request: Request) -> list[PresetVoiceRead]:
    """音色库列表（登录用户；试听 URL 每次新鲜签名不过期）。"""
    from app.middleware.jwt import get_request_user_id
    if get_request_user_id(request) is None:
        raise HTTPException(status_code=401, detail="not authenticated")

    from app.services.storage import StorageService
    st = StorageService()

    with SessionLocal() as db:
        rows = db.execute(text("""
            select voice_id, display_name, gender, age_band, style, accent, sample_url
            from preset_voices
            where is_active = true
            order by sort_order
        """)).fetchall()

    result = []
    for r in rows:
        # 从已存的 sample_url 提取 MinIO key，重新签发新鲜 presigned URL
        url = r[6]
        fresh_url = url
        if url:
            try:
                parsed = urlparse(url)
                # path 格式：/bucket/key?signature...
                path = parsed.path.lstrip("/")
                # 去掉 bucket 名（第一个路径段）
                parts = path.split("/", 1)
                if len(parts) == 2:
                    key = parts[1]
                    fresh_url = st.presigned_url(key)
            except Exception:
                fresh_url = url  # 解析失败退回原 URL
        result.append(PresetVoiceRead(
            voice_id=r[0], display_name=r[1], gender=r[2],
            age_band=r[3], style=r[4], accent=r[5],
            sample_url=fresh_url,
        ))
    return result
