from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from app.core.database import KeyframePackageRecord, KeyframeTaskRecord, SessionLocal
from app.schemas.keyframe import KeyframePackageRead, KeyframeTaskRead
from app.services.storage import storage_service


def _record_to_task(record: KeyframeTaskRecord) -> KeyframeTaskRead:
    package = KeyframePackageRead(**record.package) if record.package else None
    if package is not None:
        package = package.model_copy(
            update={"download_url": storage_service.presigned_url(package.storage_key)}
        )
    return KeyframeTaskRead(
        task_id=record.task_id,
        digital_human_id=record.digital_human_id,
        owner_user_id=record.owner_user_id,
        avatar_id=record.avatar_id,
        status=record.status,
        error=record.error,
        package=package,
        stage=record.stage,
        progress=record.progress or 0,
        started_at=record.started_at,
        finished_at=record.finished_at,
        retry_count=record.retry_count or 0,
        build_report=record.build_report,
    )


class KeyframeRepository:
    def save_task(self, task: KeyframeTaskRead) -> None:
        with SessionLocal() as db:
            record = db.get(KeyframeTaskRecord, task.task_id)
            package_data = task.package.model_dump() if task.package else None
            if record is None:
                record = KeyframeTaskRecord(
                    task_id=task.task_id,
                    digital_human_id=task.digital_human_id,
                    owner_user_id=task.owner_user_id,
                    avatar_id=task.avatar_id,
                    status=task.status,
                    error=task.error,
                    package=package_data,
                    stage=task.stage,
                    progress=task.progress,
                    started_at=task.started_at,
                    finished_at=task.finished_at,
                    retry_count=task.retry_count,
                    idempotency_key=task.idempotency_key,
                    build_report=task.build_report,
                )
                db.add(record)
            else:
                record.digital_human_id = task.digital_human_id
                record.owner_user_id = task.owner_user_id
                record.avatar_id = task.avatar_id
                record.status = task.status
                record.error = task.error
                record.package = package_data
                record.stage = task.stage
                record.progress = task.progress
                record.started_at = task.started_at
                record.finished_at = task.finished_at
                record.retry_count = task.retry_count
                record.build_report = task.build_report
            db.commit()

    def save_package(self, package: KeyframePackageRead) -> None:
        with SessionLocal() as db:
            record = db.get(KeyframePackageRecord, package.avatar_id)
            if record is None:
                record = KeyframePackageRecord(
                    avatar_id=package.avatar_id,
                    digital_human_id=package.digital_human_id,
                    version=package.version,
                    engine=package.engine,
                    package_format_version=package.package_format_version,
                    inference_width=package.inference_width,
                    inference_height=package.inference_height,
                    canvas_width=package.canvas_width,
                    canvas_height=package.canvas_height,
                    canvas_profile=package.canvas_profile,
                    renderer_mode=package.renderer_mode,
                    package_schema_revision=package.package_schema_revision,
                    renderer_runtime_version=package.renderer_runtime_version,
                    build_fingerprint=package.build_fingerprint,
                    source_asset_sha256=package.source_asset_sha256,
                    manifest_metadata=package.manifest_metadata,
                    validation_report=package.validation_report,
                    reference_render_storage_key=package.reference_render_storage_key,
                    size_bytes=package.size_bytes,
                    storage_key=package.storage_key,
                    sha256=package.sha256,
                    quality_status=package.quality_status,
                )
                db.add(record)
            else:
                record.digital_human_id = package.digital_human_id
                record.version = package.version
                record.engine = package.engine
                record.package_format_version = package.package_format_version
                record.inference_width = package.inference_width
                record.inference_height = package.inference_height
                record.canvas_width = package.canvas_width
                record.canvas_height = package.canvas_height
                record.canvas_profile = package.canvas_profile
                record.renderer_mode = package.renderer_mode
                record.package_schema_revision = package.package_schema_revision
                record.renderer_runtime_version = package.renderer_runtime_version
                record.build_fingerprint = package.build_fingerprint
                record.source_asset_sha256 = package.source_asset_sha256
                record.manifest_metadata = package.manifest_metadata
                record.validation_report = package.validation_report
                record.reference_render_storage_key = package.reference_render_storage_key
                record.size_bytes = package.size_bytes
                record.storage_key = package.storage_key
                record.sha256 = package.sha256
            db.commit()

    def get_task(self, task_id: str) -> KeyframeTaskRead | None:
        with SessionLocal() as db:
            record = db.get(KeyframeTaskRecord, task_id)
            if record is None:
                return None
            return _record_to_task(record)

    def find_by_idempotency_key(self, key: str) -> KeyframeTaskRead | None:
        """幂等查询：同一 idempotency_key 的最近一次任务。"""
        with SessionLocal() as db:
            stmt = (
                select(KeyframeTaskRecord)
                .where(KeyframeTaskRecord.idempotency_key == key)
                .order_by(KeyframeTaskRecord.created_at.desc())
                .limit(1)
            )
            record = db.execute(stmt).scalar_one_or_none()
            if record is None:
                return None
            return _record_to_task(record)

    def find_stale_running(self, older_than_seconds: int) -> list[KeyframeTaskRead]:
        """找出 running 但 started_at 早于阈值（或 started_at NULL 且 created_at 早于阈值）的任务。

        Worker 启动时调用，把这些任务重新入队或标记失败。
        """
        cutoff = datetime.now(UTC) - timedelta(seconds=older_than_seconds)
        with SessionLocal() as db:
            stmt = (
                select(KeyframeTaskRecord)
                .where(KeyframeTaskRecord.status.in_(["running", "queued"]))
                .where(
                    (KeyframeTaskRecord.started_at.is_(None) & (KeyframeTaskRecord.created_at < cutoff))
                    | (KeyframeTaskRecord.started_at < cutoff)
                )
            )
            records = db.execute(stmt).scalars().all()
            return [_record_to_task(r) for r in records]

    def update_progress(self, task_id: str, stage: str, progress: int) -> None:
        with SessionLocal() as db:
            record = db.get(KeyframeTaskRecord, task_id)
            if record is None:
                return
            record.stage = stage
            record.progress = max(0, min(100, int(progress)))
            db.commit()

    def mark_started(self, task_id: str) -> None:
        with SessionLocal() as db:
            record = db.get(KeyframeTaskRecord, task_id)
            if record is None:
                return
            record.status = "running"
            record.stage = "preparing"
            record.started_at = datetime.now(UTC)
            db.commit()

    def mark_finished(
        self,
        task_id: str,
        *,
        status: str,
        error: str | None = None,
        report: dict | None = None,
        stage: str | None = None,
    ) -> None:
        with SessionLocal() as db:
            record = db.get(KeyframeTaskRecord, task_id)
            if record is None:
                return
            record.status = status
            record.error = error
            record.build_report = report
            record.finished_at = datetime.now(UTC)
            if stage is not None:
                record.stage = stage
            if status == "completed":
                record.progress = 100
            db.commit()

    def increment_retry(self, task_id: str) -> int:
        """返回当前 retry_count（自增后）。"""
        with SessionLocal() as db:
            record = db.get(KeyframeTaskRecord, task_id)
            if record is None:
                return 0
            record.retry_count = (record.retry_count or 0) + 1
            db.commit()
            return record.retry_count

    def get_package(self, avatar_id: str) -> KeyframePackageRead | None:
        with SessionLocal() as db:
            record = db.get(KeyframePackageRecord, avatar_id)
            if record is None:
                return None
            if record.quality_status == "rejected":
                return None
            return KeyframePackageRead(
                avatar_id=record.avatar_id,
                digital_human_id=record.digital_human_id,
                version=record.version,
                engine=record.engine,
                package_format_version=record.package_format_version,
                inference_width=record.inference_width,
                inference_height=record.inference_height,
                canvas_width=record.canvas_width,
                canvas_height=record.canvas_height,
                canvas_profile=record.canvas_profile,
                renderer_mode=record.renderer_mode,
                package_schema_revision=record.package_schema_revision,
                renderer_runtime_version=record.renderer_runtime_version,
                build_fingerprint=record.build_fingerprint,
                source_asset_sha256=record.source_asset_sha256,
                manifest_metadata=record.manifest_metadata,
                validation_report=record.validation_report,
                reference_render_storage_key=record.reference_render_storage_key,
                size_bytes=record.size_bytes,
                storage_key=record.storage_key,
                download_url=storage_service.presigned_url(record.storage_key),
                sha256=record.sha256,
                quality_status=record.quality_status,
            )

    def get_latest_package(self, digital_human_id: str) -> KeyframePackageRead | None:
        with SessionLocal() as db:
            stmt = (
                select(KeyframePackageRecord)
                .where(KeyframePackageRecord.digital_human_id == digital_human_id)
                .order_by(KeyframePackageRecord.created_at.desc())
                .limit(1)
            )
            record = db.execute(stmt).scalar_one_or_none()
            if record is None:
                return None
            return self.get_package(record.avatar_id)

    def get_package_by_storage_key(self, storage_key: str) -> KeyframePackageRead | None:
        with SessionLocal() as db:
            record = db.execute(
                select(KeyframePackageRecord)
                .where(KeyframePackageRecord.storage_key == storage_key)
                .limit(1)
            ).scalar_one_or_none()
            if record is None:
                return None
            avatar_id = record.avatar_id
        return self.get_package(avatar_id)


keyframe_repository = KeyframeRepository()
