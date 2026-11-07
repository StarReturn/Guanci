import hashlib
import shutil
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import boto3
from botocore.client import Config

from app.core.config import settings


class StorageService:
    def __init__(self) -> None:
        self._root = settings.local_storage_root

    def put_bytes(self, key: str, content: bytes, content_type: str) -> tuple[str, int, str]:
        sha256 = hashlib.sha256(content).hexdigest()
        if settings.storage_provider == "s3":
            client = self._s3_client()
            client.put_object(
                Bucket=settings.storage_bucket,
                Key=key,
                Body=content,
                ContentType=content_type,
            )
        else:
            path = self._root / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        return key, len(content), sha256

    def get_bytes(self, key: str) -> bytes:
        """Fetch raw bytes for ``key``. Used by the async worker to reload
        the source image uploaded by the API container.

        Raises ``FileNotFoundError`` if the object does not exist in either
        backend. S3 errors surface as ``botocore.ClientError``.
        """
        if settings.storage_provider == "s3":
            client = self._s3_client()
            resp = client.get_object(Bucket=settings.storage_bucket, Key=key)
            return resp["Body"].read()
        path = self._root / key
        if not path.is_file():
            raise FileNotFoundError(f"storage object not found: {key}")
        return path.read_bytes()

    def presigned_url(self, key: str) -> str:
        if settings.storage_provider == "s3":
            client = self._s3_client(
                endpoint_url=settings.storage_public_endpoint_url
                or settings.storage_endpoint_url
            )
            return client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.storage_bucket, "Key": key},
                ExpiresIn=settings.storage_presigned_url_expires_seconds,
            )
        encoded_key = quote(key, safe="")
        return f"{settings.public_base_url.rstrip('/')}{settings.api_prefix}/keyframes/packages/download?key={encoded_key}"

    def resolve_local_path(self, key: str) -> Path | None:
        if settings.storage_provider != "local":
            return None
        root = self._root.resolve()
        path = (root / key).resolve()
        if path != root and root not in path.parents:
            return None
        return path

    def playback_url(self, reference: str | None) -> str | None:
        """Resolve durable message references and refresh legacy URLs owned by this store."""
        if not reference:
            return None
        if reference.startswith("storage://"):
            key = reference[len("storage://"):]
        else:
            parsed = urlparse(reference)
            endpoints = (settings.storage_public_endpoint_url, settings.storage_endpoint_url)
            owned_hosts = {urlparse(endpoint).netloc for endpoint in endpoints if endpoint}
            path = unquote(parsed.path).lstrip("/")
            prefix = settings.storage_bucket + "/"
            if parsed.netloc in owned_hosts and path.startswith(prefix):
                key = path[len(prefix):]
            elif (
                parsed.netloc == urlparse(settings.public_base_url).netloc
                and parsed.path.endswith("/keyframes/packages/download")
            ):
                key = parse_qs(parsed.query).get("key", [""])[0]
            else:
                return reference
        if not key.startswith(("digital_humans/", "dialogue/")) or ".." in key.split("/"):
            return None
        return self.presigned_url(key)

    def delete_key(self, key: str) -> None:
        if settings.storage_provider == "s3":
            self._s3_client().delete_object(Bucket=settings.storage_bucket, Key=key)
            return
        path = self.resolve_local_path(key)
        if path is not None and path.is_file():
            path.unlink()

    def delete_prefix(self, prefix: str) -> int:
        """Delete every object below an application-owned prefix."""
        normalized = prefix.strip("/") + "/"
        if settings.storage_provider == "s3":
            client = self._s3_client()
            continuation_token = None
            deleted = 0
            while True:
                params = {"Bucket": settings.storage_bucket, "Prefix": normalized}
                if continuation_token:
                    params["ContinuationToken"] = continuation_token
                response = client.list_objects_v2(**params)
                objects = [{"Key": item["Key"]} for item in response.get("Contents", [])]
                if objects:
                    client.delete_objects(
                        Bucket=settings.storage_bucket,
                        Delete={"Objects": objects, "Quiet": True},
                    )
                    deleted += len(objects)
                if not response.get("IsTruncated"):
                    return deleted
                continuation_token = response.get("NextContinuationToken")

        path = self.resolve_local_path(normalized)
        if path is None or not path.exists():
            return 0
        count = sum(1 for item in path.rglob("*") if item.is_file())
        shutil.rmtree(path)
        return count

    def _s3_client(self, *, endpoint_url: str | None = None):
        return boto3.client(
            "s3",
            endpoint_url=endpoint_url or settings.storage_endpoint_url,
            aws_access_key_id=settings.storage_access_key,
            aws_secret_access_key=settings.storage_secret_key,
            region_name=settings.storage_region,
            use_ssl=settings.storage_use_ssl,
            config=Config(s3={"addressing_style": "path"}),
        )


storage_service = StorageService()
