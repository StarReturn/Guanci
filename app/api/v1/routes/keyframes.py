from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.middleware.jwt import get_request_user_id
from app.schemas.keyframe import KeyframeBuildResponse, KeyframePackageRead, KeyframeTaskRead
from app.services.auth_service import quota_service
from app.services.keyframe_builder import keyframe_builder
from app.services.storage import storage_service

router = APIRouter()


@router.post("/build", response_model=KeyframeBuildResponse)
async def build_keyframe_package(
    digital_human_id: str,
) -> KeyframeBuildResponse:
    raise HTTPException(
        status_code=422,
        detail=(
            "legacy photo build routes removed (2026-08-25); the user entry "
            "is photo-only: /api/v1/me/digital-humans/{id}/photo-preflight "
            "then /{id}/build-dhlive (video upload closed by product decision)"
        ),
    )


@router.get("/tasks/{task_id}", response_model=KeyframeTaskRead)
async def get_keyframe_task(task_id: str) -> KeyframeTaskRead:
    task = keyframe_builder.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task


@router.get("/packages/download")
async def download_local_package(request: Request, key: str) -> FileResponse:
    _ = request
    # Stats only (decision 2026-06-23)
    uid = get_request_user_id(request)
    path = storage_service.resolve_local_path(key)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="package not found")
    if uid:
        try:
            quota_service.consume(uid, "download")
        except Exception:
            pass
    return FileResponse(path, media_type="application/zip", filename=path.name)


@router.get("/packages/{avatar_id}", response_model=KeyframePackageRead)
async def get_keyframe_package(request: Request, avatar_id: str) -> KeyframePackageRead:
    # `request` is required by slowapi for rate-limit injection.
    _ = request
    package = keyframe_builder.get_package(avatar_id)
    if package is None:
        raise HTTPException(status_code=404, detail="package not found")
    return package
