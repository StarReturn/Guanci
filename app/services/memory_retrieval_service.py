"""Hybrid retrieval for the relationship-scoped memory space.

Pipeline:

1. Relationship-space pruning: ``user_id + digital_human_id + status=active
   + sensitivity=normal + not expired``. Sensitive rows are never selected.
2. Optional dense recall via pgvector cosine against the embedding service.
3. Lexical recall via pg_trgm similarity against ``search_text``.
4. Hard threshold floor: any candidate below both thresholds is dropped,
   regardless of importance or top-K demand.
5. Deterministic weighted score (dense 0.45 + lexical 0.15 + importance
   0.15 + confidence 0.10 + recency 0.10 + user_confirmed 0.05).
6. Top-K by score, returned with a per-item score breakdown for the
   injection log.

Degradation contract:

* If the embedding service is unavailable, we run lexical-only with the
  same hard threshold. ``fallback_reason='embedding_unavailable'`` is
  recorded on the injection log.
* If lexical itself returns nothing we return an EMPTY list. We never
  backfill with importance-ranked rows.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import or_, select, text, update

from app.core.config import settings
from app.core.database import MemoryInjectionLogRecord, MemoryItemRecord, SessionLocal
from app.services.embedding_client import EmbeddingUnavailable, embedding_client
from app.services.entitlement_stub import EffectiveMembership

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)


def _normalise(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", (value or "").lower())


def _lexical_tokens(value: str) -> set[str]:
    normalised = _normalise(value)
    cjk = "".join(c for c in normalised if "\u4e00" <= c <= "\u9fff")
    tokens = {cjk[i:i + 2] for i in range(max(0, len(cjk) - 1))}
    tokens.update(re.findall(r"[a-z0-9]{2,}", value.lower()))
    tokens.discard("")
    return tokens


@dataclass(frozen=True)
class ScoredItem:
    record: MemoryItemRecord
    score: float
    dense_similarity: float
    lexical_similarity: float
    importance: float
    confidence: float
    recency: float
    user_confirmed: float


@dataclass
class RetrievalResult:
    items: list[ScoredItem]
    retrieval_mode: str  # hybrid | lexical_only | empty
    candidate_count: int
    latency_ms: int
    fallback_reason: str | None
    embedding_revision: str | None


class MemoryRetrievalService:
    def retrieve(
        self, *, user_id: str, digital_human_id: str, session_id: str,
        query: str, membership: EffectiveMembership, limit: int | None = None,
    ) -> RetrievalResult:
        if not digital_human_id:
            return RetrievalResult(
                items=[], retrieval_mode="empty",
                candidate_count=0, latency_ms=0,
                fallback_reason="missing_digital_human_id",
                embedding_revision=None,
            )
        if not bool(membership.entitlements.get("cross_session_memory", False)):
            return RetrievalResult(
                items=[], retrieval_mode="empty",
                candidate_count=0, latency_ms=0,
                fallback_reason="plan_disables_cross_session_memory",
                embedding_revision=None,
            )
        top_k = int(limit or membership.entitlements.get("memory_recall_count", 0))
        if top_k <= 0:
            return RetrievalResult(
                items=[], retrieval_mode="empty",
                candidate_count=0, latency_ms=0,
                fallback_reason="plan_recall_count_zero",
                embedding_revision=None,
            )
        started = time.monotonic()
        query_text = (query or "").strip()
        if not query_text:
            return RetrievalResult(
                items=[], retrieval_mode="empty",
                candidate_count=0, latency_ms=0,
                fallback_reason="empty_query",
                embedding_revision=None,
            )

        # Try embedding first; if it fails we degrade to lexical-only.
        embedding_revision: str | None = None
        dense_pairs: list[tuple[str, float]] = []
        fallback_reason: str | None = None
        retrieval_mode = "hybrid"
        try:
            embedding_result = _safe_sync_embed(query_text)
            if embedding_result is not None:
                embedding_revision = embedding_result.revision
                dense_pairs = self._dense_recall(
                    user_id=user_id, digital_human_id=digital_human_id,
                    vector=embedding_result.vector, top_k=top_k * 4,
                )
        except EmbeddingUnavailable as exc:
            fallback_reason = "embedding_unavailable"
            retrieval_mode = "lexical_only"
            logger.info("memory_embedding_unavailable detail=%s", str(exc)[:160])

        lexical_pairs = self._lexical_recall(
            user_id=user_id, digital_human_id=digital_human_id,
            query=query_text, top_k=top_k * 4,
        )

        scored = self._merge_and_score(
            user_id=user_id, digital_human_id=digital_human_id,
            dense_pairs=dense_pairs, lexical_pairs=lexical_pairs,
            now=_now(),
        )

        # Hard threshold: keep a candidate only if it cleared at least one
        # of the two floors. Importance can never substitute.
        floor_dense = float(settings.memory_dense_threshold)
        floor_lexical = float(settings.memory_lexical_threshold)
        kept = [
            s for s in scored
            if s.dense_similarity >= floor_dense or s.lexical_similarity >= floor_lexical
        ]
        kept.sort(key=lambda s: s.score, reverse=True)
        selected = kept[:top_k]

        latency_ms = int((time.monotonic() - started) * 1000)
        self._log_injection(
            user_id=user_id, digital_human_id=digital_human_id,
            session_id=session_id, query=query_text,
            selected=selected, retrieval_mode=retrieval_mode,
            candidate_count=len(scored), latency_ms=latency_ms,
            fallback_reason=fallback_reason,
            embedding_revision=embedding_revision,
            membership=membership,
        )
        return RetrievalResult(
            items=selected, retrieval_mode=retrieval_mode,
            candidate_count=len(scored), latency_ms=latency_ms,
            fallback_reason=fallback_reason, embedding_revision=embedding_revision,
        )

    # ------------------------------------------------------------------ dense
    def _dense_recall(
        self, *, user_id: str, digital_human_id: str,
        vector: list[float], top_k: int,
    ) -> list[tuple[str, float]]:
        # pgvector cosine distance. We materialise the vector as a string
        # literal so SQLAlchemy does not need the pgvector adapter at write
        # time — this is the same shape the adapter emits.
        vec_literal = "[" + ",".join(f"{x:.7f}" for x in vector) + "]"
        sql = text(
            """
            SELECT memory_id,
                   1 - (embedding <=> CAST(:vec AS vector)) AS dense_similarity
            FROM memory_items
            WHERE user_id = :uid
              AND digital_human_id = :dhid
              AND status = 'active'
              AND sensitivity = 'normal'
              AND embedding IS NOT NULL
              AND (expires_at IS NULL OR expires_at > now())
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :k
            """
        )
        with SessionLocal() as db:
            rows = db.execute(sql, {
                "uid": user_id, "dhid": digital_human_id,
                "vec": vec_literal, "k": top_k,
            }).all()
        return [(str(r[0]), float(r[1] or 0.0)) for r in rows]

    # ---------------------------------------------------------------- lexical
    def _lexical_recall(
        self, *, user_id: str, digital_human_id: str,
        query: str, top_k: int,
    ) -> list[tuple[str, float]]:
        sql = text(
            """
            SELECT memory_id,
                   similarity(search_text, :q) AS lexical_similarity
            FROM memory_items
            WHERE user_id = :uid
              AND digital_human_id = :dhid
              AND status = 'active'
              AND sensitivity = 'normal'
              AND search_text IS NOT NULL
              AND (expires_at IS NULL OR expires_at > now())
              AND search_text % :q
            ORDER BY similarity(search_text, :q) DESC
            LIMIT :k
            """
        )
        with SessionLocal() as db:
            rows = db.execute(sql, {
                "uid": user_id, "dhid": digital_human_id,
                "q": query, "k": top_k,
            }).all()
        return [(str(r[0]), float(r[1] or 0.0)) for r in rows]

    # ------------------------------------------------------------------ merge
    def _merge_and_score(
        self, *, user_id: str, digital_human_id: str,
        dense_pairs: list[tuple[str, float]],
        lexical_pairs: list[tuple[str, float]],
        now: datetime,
    ) -> list[ScoredItem]:
        candidate_ids = {mid for mid, _ in dense_pairs} | {mid for mid, _ in lexical_pairs}
        if not candidate_ids:
            return []
        with SessionLocal() as db:
            rows = db.execute(
                select(MemoryItemRecord).where(
                    MemoryItemRecord.user_id == user_id,
                    MemoryItemRecord.digital_human_id == digital_human_id,
                    MemoryItemRecord.memory_id.in_(candidate_ids),
                    MemoryItemRecord.status == "active",
                    MemoryItemRecord.sensitivity == "normal",
                    or_(
                        MemoryItemRecord.expires_at.is_(None),
                        MemoryItemRecord.expires_at > now,
                    ),
                )
            ).scalars().all()
        dense_map = {mid: score for mid, score in dense_pairs}
        lexical_map = {mid: score for mid, score in lexical_pairs}
        scored: list[ScoredItem] = []
        for row in rows:
            dense = max(0.0, min(1.0, dense_map.get(row.memory_id, 0.0)))
            lexical = max(0.0, min(1.0, lexical_map.get(row.memory_id, 0.0)))
            importance = max(0.0, min(1.0, float(row.importance_score or 0.0)))
            confidence = max(0.0, min(1.0, float(row.confidence_score or 0.0)))
            age_days = max(0, (now - (row.updated_at or row.created_at or now)).days)
            recency = 1.0 / (1.0 + age_days / 30.0)
            confirmed = 1.0 if row.user_confirmed else 0.0
            score = (
                dense * float(settings.memory_dense_weight)
                + lexical * float(settings.memory_lexical_weight)
                + importance * float(settings.memory_importance_weight)
                + confidence * float(settings.memory_confidence_weight)
                + recency * float(settings.memory_recency_weight)
                + confirmed * float(settings.memory_user_confirmed_weight)
            )
            scored.append(ScoredItem(
                record=row, score=score,
                dense_similarity=dense, lexical_similarity=lexical,
                importance=importance, confidence=confidence,
                recency=recency, user_confirmed=confirmed,
            ))
        return scored

    # -------------------------------------------------------------------- log
    def _log_injection(
        self, *, user_id: str, digital_human_id: str, session_id: str,
        query: str, selected: list[ScoredItem], retrieval_mode: str,
        candidate_count: int, latency_ms: int, fallback_reason: str | None,
        embedding_revision: str | None, membership: EffectiveMembership,
    ) -> None:
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        memory_ids = [s.record.memory_id for s in selected]
        scores = [
            {
                "memory_id": s.record.memory_id,
                "score": round(s.score, 4),
                "dense": round(s.dense_similarity, 4),
                "lexical": round(s.lexical_similarity, 4),
                "importance": round(s.importance, 4),
                "confidence": round(s.confidence, 4),
                "recency": round(s.recency, 4),
                "user_confirmed": round(s.user_confirmed, 4),
            }
            for s in selected
        ]
        with SessionLocal() as db:
            db.add(MemoryInjectionLogRecord(
                injection_id=str(uuid4()),
                user_id=user_id, digital_human_id=digital_human_id,
                session_id=session_id, query_hash=query_hash,
                memory_ids=memory_ids, injected_count=len(memory_ids),
                policy_plan_code=membership.plan_code,
                retrieval_mode=retrieval_mode,
                candidate_count=candidate_count,
                selected_scores=scores,
                embedding_revision=embedding_revision,
                latency_ms=latency_ms,
                fallback_reason=fallback_reason,
            ))
            if selected:
                # Scored records were loaded in a different Session and are
                # detached here. Update by scoped IDs so counters persist.
                db.execute(update(MemoryItemRecord).where(
                    MemoryItemRecord.user_id == user_id,
                    MemoryItemRecord.digital_human_id == digital_human_id,
                    MemoryItemRecord.memory_id.in_(memory_ids),
                ).values(
                    recall_count=MemoryItemRecord.recall_count + 1,
                    last_recalled_at=_now(),
                ))
            db.commit()


def _safe_sync_embed(query_text: str):
    """Run the async embedding client from a sync context.

    The retrieval service is invoked synchronously by
    ``DialogueService.build_llm_messages`` which itself is called from an
    async FastAPI handler. We use a private event loop to avoid
    interfering with the caller's loop on Windows proactor event loops.
    """
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(embedding_client.embed_one(query_text))
    # We are inside an active loop: schedule the coroutine via a worker
    # thread that runs its own loop. This is rare (only when retrieval is
    # called from inside an async caller); the normal caller is sync.
    import threading

    box: dict = {}

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            box["result"] = loop.run_until_complete(
                embedding_client.embed_one(query_text)
            )
        except Exception as exc:
            box["error"] = exc
        finally:
            loop.close()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["result"]


memory_retrieval_service = MemoryRetrievalService()


__all__ = [
    "MemoryRetrievalService",
    "RetrievalResult",
    "ScoredItem",
    "memory_retrieval_service",
]
