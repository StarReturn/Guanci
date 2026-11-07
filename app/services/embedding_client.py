"""HTTP client for the standalone CPU embedding service.

Contract:

    POST {endpoint}/v1/embeddings
        {"inputs": ["text1", "text2", ...]}
    ->  {"data": [
            {"embedding": [...], "model": "BAAI/bge-small-zh-v1.5", "revision": "v1.0.0", "dim": 512},
            ...
        ]}

The client validates model_name / revision / dim against ``settings`` and
against the active row in ``embedding_provider_configs``. Any mismatch is a
hard failure (``EmbeddingUnavailable``) — we never silently fall back.

Fail-closed behaviour matters more than availability here: if the embedding
container is down or returns the wrong model, ``retrieve`` falls back to
strict-lexical with a hard threshold (NOT importance-ranked), or returns an
empty result set. The writer leaves new memory_items without an embedding
and schedules a ``reembed`` job; once the service recovers, the worker
back-fills vectors and the dense path comes back online.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select

from app.core.config import settings
from app.core.database import EmbeddingProviderConfigRecord, SessionLocal
from app.core.http_client import build_resilient_client

logger = logging.getLogger(__name__)


class EmbeddingUnavailable(RuntimeError):
    """Raised when the embedding service cannot be used safely.

    The caller MUST treat this as "no dense retrieval available". Recovery
    options:

    * retrieval -> strict-lexical-only or empty.
    * writer -> skip embedding, schedule a reembed job.
    """


@dataclass(frozen=True)
class EmbeddingResult:
    vector: list[float]
    model: str
    revision: str
    dim: int


class EmbeddingClient:
    def _active_config(self) -> EmbeddingProviderConfigRecord | None:
        with SessionLocal() as db:
            return db.execute(
                select(EmbeddingProviderConfigRecord).where(
                    EmbeddingProviderConfigRecord.is_active.is_(True)
                ).limit(1)
            ).scalar_one_or_none()

    def _endpoint(self, db_config: EmbeddingProviderConfigRecord | None) -> str:
        if db_config is not None:
            return db_config.endpoint
        return settings.memory_embedding_endpoint

    def _contract(
        self, db_config: EmbeddingProviderConfigRecord | None
    ) -> tuple[str, str, int]:
        """Resolve the (model, revision, dim) contract.

        Order of precedence:

        1. Active row in ``embedding_provider_configs`` (operator override).
        2. Settings defaults.

        We never accept whatever the server reports; the server MUST agree
        with our contract.
        """
        if db_config is not None:
            return db_config.model_name, db_config.revision, db_config.dim
        return (
            settings.memory_embedding_required_model,
            settings.memory_embedding_required_revision,
            settings.memory_embedding_required_dim,
        )

    async def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        if not texts:
            return []
        db_config = self._active_config()
        endpoint = self._endpoint(db_config)
        expected_model, expected_revision, expected_dim = self._contract(db_config)

        timeout = max(3, int(settings.memory_embedding_timeout_seconds))
        try:
            async with build_resilient_client(read_timeout=timeout) as client:
                resp = await client.post(
                    f"{endpoint.rstrip('/')}/v1/embeddings",
                    json={"inputs": list(texts)},
                    headers={"Content-Type": "application/json"},
                )
        except Exception as exc:  # network / DNS / connection refused
            raise EmbeddingUnavailable(
                f"embedding service unreachable at {endpoint}: {exc}"
            ) from exc
        if resp.status_code != 200:
            raise EmbeddingUnavailable(
                f"embedding service returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        try:
            body = resp.json()
        except Exception as exc:
            raise EmbeddingUnavailable("embedding service returned non-JSON body") from exc

        data = body.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise EmbeddingUnavailable(
                f"embedding service returned malformed payload "
                f"(expected {len(texts)} items, got {len(data) if isinstance(data, list) else 'non-list'})"
            )
        results: list[EmbeddingResult] = []
        for row in data:
            model = str(row.get("model") or "")
            revision = str(row.get("revision") or "")
            dim = int(row.get("dim") or 0)
            vector = row.get("embedding")
            if (
                model != expected_model
                or revision != expected_revision
                or dim != expected_dim
                or not isinstance(vector, list)
                or len(vector) != expected_dim
            ):
                raise EmbeddingUnavailable(
                    "embedding service contract mismatch: "
                    f"got model={model!r} revision={revision!r} dim={dim} len={len(vector) if isinstance(vector, list) else 0}, "
                    f"expected model={expected_model!r} revision={expected_revision!r} dim={expected_dim}"
                )
            results.append(EmbeddingResult(
                vector=[float(x) for x in vector],
                model=model, revision=revision, dim=dim,
            ))
        return results

    async def embed_one(self, text: str) -> EmbeddingResult:
        results = await self.embed([text])
        return results[0]


embedding_client = EmbeddingClient()


__all__ = ["EmbeddingClient", "EmbeddingResult", "EmbeddingUnavailable", "embedding_client"]
