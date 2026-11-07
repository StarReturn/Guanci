"""Admin routes for background management."""
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.notification_service import background_service

router = APIRouter()


class BackgroundRead(BaseModel):
    background_id: str
    name: str
    source: str


@router.get("", response_model=list[BackgroundRead])
async def list_all() -> list[BackgroundRead]:
    return [
        BackgroundRead(background_id=b.background_id, name=b.name, source=b.source)
        for b in background_service.list()
    ]


@router.post("", response_model=BackgroundRead, status_code=201)
async def upload(name: str, file: UploadFile = File(...)) -> BackgroundRead:
    content = await file.read()
    rec = background_service.create(
        name=name, content=content,
        content_type=file.content_type or "image/jpeg",
        source="preset",
    )
    return BackgroundRead(background_id=rec.background_id, name=rec.name, source=rec.source)


@router.delete("/{background_id}")
async def delete(background_id: str) -> dict:
    ok = background_service.delete(background_id)
    if not ok:
        raise HTTPException(status_code=404, detail="not found")
    return {"deleted": True}
