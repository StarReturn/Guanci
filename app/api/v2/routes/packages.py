from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.middleware.jwt import get_request_user_id
from app.services.digital_human_service import DigitalHumanNotFound, digital_human_service
from app.services.storage import storage_service


router = APIRouter()


class DhLivePackageDescriptor(BaseModel):
    package_version_id: str
    digital_human_id: str
    avatar_id: str
    version: int
    engine: str
    renderer_mode: str
    package_format_version: int
    package_schema_revision: int
    renderer_runtime_version: str
    build_fingerprint: str
    size_bytes: int
    sha256: str
    download_url: str


@router.get(
    "/digital-humans/{digital_human_id}/package",
    response_model=DhLivePackageDescriptor,
)
async def get_active_dhlive_package(
    request: Request, digital_human_id: str
) -> DhLivePackageDescriptor:
    user_id = get_request_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="authentication required")
    try:
        digital_human_service.get_owned(user_id, digital_human_id)
    except DigitalHumanNotFound as exc:
        raise HTTPException(status_code=404, detail="digital human not found") from exc
    package = digital_human_service.get_active_package(digital_human_id)
    if package is None:
        raise HTTPException(status_code=409, detail="active package is not ready")
    if (
        package.engine != "dhlive"
        or package.renderer_mode != "dhlive_webview_v1"
        or package.package_format_version != 1
        or package.package_schema_revision != 1
        or not package.renderer_runtime_version
        or not package.build_fingerprint
        or not package.validation_report
        or package.validation_report.get("passed") is not True
        or package.validation_report.get("reference_render", {}).get("passed") is not True
    ):
        raise HTTPException(status_code=409, detail="active package is not DH_live-ready")
    return DhLivePackageDescriptor(
        package_version_id=package.package_version_id,
        digital_human_id=package.digital_human_id,
        avatar_id=package.avatar_id,
        version=package.version,
        engine=package.engine,
        renderer_mode=package.renderer_mode,
        package_format_version=package.package_format_version,
        package_schema_revision=package.package_schema_revision,
        renderer_runtime_version=package.renderer_runtime_version,
        build_fingerprint=package.build_fingerprint,
        size_bytes=package.size_bytes,
        sha256=package.sha256,
        download_url=storage_service.presigned_url(package.storage_key),
    )
