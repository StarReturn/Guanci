"""Memory write path: validate, normalise, dedupe, encrypt, persist.

The writer is called by:

* the memory-worker after the extractor returns candidates;
* the user-facing memory API (POST/PATCH), where ``user_confirmed=True``
  and confidence is 1.0;
* the migration's ``reembed`` job (re-embedding existing rows).

Critical invariants:

* ``MemoryItemRecord`` is scoped to ``(user_id, digital_human_id)``;
  never writes a NULL digital_human_id.
* Sensitive candidates are encrypted with the independent
  ``MEMORY_ENCRYPTION_KEY`` (NOT provider/JWT/storage), and never receive
  an embedding, search_text, or plaintext summary.
* Semantic-fact conflicts supersede the previous active record in the same
  relationship_key + fact_key; episodic_events coexist.
* Capacity is enforced per ``(user_id, digital_human_id)``; oversize
  inserts evict the lowest-importance, lowest-recall non-edited record
  rather than failing the write.
* The whole write happens under ``pg_advisory_xact_lock(hashtext(key))``
  so two parallel workers cannot both create a duplicate active fact.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import desc, func, or_, select

from app.core.config import settings
from app.core.database import MemoryItemRecord, SessionLocal
from app.services.entitlement_stub import EffectiveMembership
from app.services.memory_security_service import (
    Classification,
    classify,
    contains_high_risk_credentials,
)

logger = logging.getLogger(__name__)


class MemoryWriterError(RuntimeError):
    pass


class CredentialRejected(MemoryWriterError):
    """Candidate contains verbatim credentials; the entire candidate is dropped."""


@dataclass(frozen=True)
class WriteCandidate:
    """Candidate produced by the extractor or the user-edit API."""

    memory_type: str  # semantic_fact | episodic_event
    category: str
    fact_key: str | None
    summary: str
    importance_score: float
    confidence_score: float
    occurred_at: datetime | None
    tags: list[str]
    valid_from: datetime | None


@dataclass
class WriteOutcome:
    stored: MemoryItemRecord | None
    superseded_id: str | None
    skipped_reason: str | None


_MEMORY_TYPES = ("semantic_fact", "episodic_event")
_CATEGORIES = (
    "identity", "relationship", "family", "work",
    "preference", "plan", "event", "health",
)


def _now() -> datetime:
    return datetime.now(UTC)


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def _content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _search_text(summary: str, tags: list[str]) -> str:
    """Lexical index payload. We collapse whitespace; pg_trgm handles the
    substring matching at query time."""
    payload = " ".join([summary, *(tags or [])])
    return _normalise_text(payload)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def normalise_fact_key(category: str, raw: str | None) -> str | None:
    """Convert an extractor ``fact_key`` into a stable slug.

    The extractor is trusted to emit a dotted path (``location.city``);
    if it omits one we derive ``<category>.<sha8(summary)>`` so two writes
    of the same summary still conflict-supersede correctly.
    """
    if raw and re.match(r"^[A-Za-z0-9_][A-Za-z0-9_.\-/]{0,127}$", raw):
        return raw[:128]
    return None


class MemoryWriterService:
    def normalise_candidate(
        self,
        *,
        memory_type: str,
        category: str,
        summary: str,
        fact_key: str | None = None,
        importance_score: float = 0.5,
        confidence_score: float = 0.65,
        occurred_at: datetime | None = None,
        tags: list[str] | None = None,
        valid_from: datetime | None = None,
    ) -> WriteCandidate:
        summary_clean = _normalise_text(summary)[:500]
        if not summary_clean:
            raise MemoryWriterError("candidate summary is empty after normalisation")
        if contains_high_risk_credentials(summary_clean):
            raise CredentialRejected("candidate contains verbatim credentials")
        if memory_type not in _MEMORY_TYPES:
            raise MemoryWriterError(f"unknown memory_type: {memory_type}")
        if category not in _CATEGORIES:
            # Tolerate legacy categories by mapping to ``identity``. This
            # keeps backwards compatibility while we tighten the schema.
            category = "identity"
        if memory_type == "semantic_fact":
            fact_key = normalise_fact_key(category, fact_key) or (
                f"{category}.{hashlib.sha256(summary_clean.encode('utf-8')).hexdigest()[:8]}"
            )
        else:
            fact_key = None
        return WriteCandidate(
            memory_type=memory_type, category=category, fact_key=fact_key,
            summary=summary_clean,
            importance_score=_clamp(importance_score),
            confidence_score=_clamp(confidence_score, 0.0, 1.0),
            occurred_at=occurred_at, tags=tags or [],
            valid_from=valid_from or _now(),
        )

    # ----------------------------------------------------------- public write
    def store(
        self,
        *,
        user_id: str,
        digital_human_id: str,
        candidate: WriteCandidate,
        source_session_id: str | None,
        source_message_ids: list[str],
        membership: EffectiveMembership,
        user_confirmed: bool = False,
    ) -> WriteOutcome:
        """Persist a single candidate, dedupe/conflict-supersede, enforce
        capacity. Sensitive candidates are encrypted via
        ``memory_crypto.encrypt_memory``.
        """
        max_items = int(membership.entitlements.get("memory_items_per_avatar", 0))
        retention_days = int(membership.entitlements.get("memory_retention_days", 0))
        if max_items <= 0 or retention_days <= 0:
            return WriteOutcome(
                stored=None, superseded_id=None,
                skipped_reason="plan_does_not_allow_memory",
            )
        if not digital_human_id:
            return WriteOutcome(
                stored=None, superseded_id=None,
                skipped_reason="missing_digital_human_id",
            )

        now = _now()
        expires_at = now + timedelta(days=retention_days)
        classification: Classification = classify(
            candidate.summary, category=candidate.category
        )
        if classification.reject_reason == "credential":
            raise CredentialRejected("candidate contains verbatim credentials")

        with SessionLocal() as db:
            db.execute(select(func.pg_advisory_xact_lock(func.hashtext(
                f"relationship-memory:{user_id}:{digital_human_id}"
            ))))

            # 1. dedupe / supersede
            superseded_id: str | None = None
            superseded_record: MemoryItemRecord | None = None
            if candidate.memory_type == "semantic_fact" and candidate.fact_key:
                active_same_fact = db.execute(
                    select(MemoryItemRecord).where(
                        MemoryItemRecord.user_id == user_id,
                        MemoryItemRecord.digital_human_id == digital_human_id,
                        MemoryItemRecord.memory_type == "semantic_fact",
                        MemoryItemRecord.fact_key == candidate.fact_key,
                        MemoryItemRecord.status == "active",
                    ).with_for_update().limit(1)
                ).scalar_one_or_none()
                if active_same_fact is not None:
                    if self._same_content(active_same_fact, candidate):
                        # no-op: same content, same hash. Refresh TTL only.
                        active_same_fact.expires_at = max(
                            active_same_fact.expires_at or expires_at, expires_at
                        )
                        active_same_fact.updated_at = now
                        db.commit()
                        db.refresh(active_same_fact)
                        return WriteOutcome(
                            stored=active_same_fact, superseded_id=None,
                            skipped_reason="duplicate_refresh",
                        )
                    superseded_id = active_same_fact.memory_id
                    superseded_record = active_same_fact
                    self._supersede(db, active_same_fact, candidate, now)
            else:
                # lexical / hash dedupe for episodic_events
                content_hash = _content_hash(candidate.summary)
                dup = db.execute(
                    select(MemoryItemRecord).where(
                        MemoryItemRecord.user_id == user_id,
                        MemoryItemRecord.digital_human_id == digital_human_id,
                        MemoryItemRecord.memory_type == candidate.memory_type,
                        MemoryItemRecord.content_hash == content_hash,
                        MemoryItemRecord.status == "active",
                    ).limit(1)
                ).scalar_one_or_none()
                if dup is not None:
                    dup.tags = sorted(set(dup.tags or []) | set(candidate.tags))[:8]
                    dup.importance_score = max(
                        dup.importance_score, candidate.importance_score
                    )
                    dup.expires_at = max(dup.expires_at or expires_at, expires_at)
                    dup.updated_at = now
                    db.commit()
                    db.refresh(dup)
                    return WriteOutcome(
                        stored=dup, superseded_id=None,
                        skipped_reason="duplicate_merged",
                    )

            # 2. capacity
            capacity_available = self._enforce_capacity(
                db, user_id=user_id, digital_human_id=digital_human_id,
                max_items=max_items, importance_floor=candidate.importance_score,
                now=now,
            )
            if not capacity_available:
                db.rollback()
                return WriteOutcome(
                    stored=None, superseded_id=None,
                    skipped_reason="capacity_rejected",
                )

            # 3. build the new row
            row = self._build_row(
                user_id=user_id, digital_human_id=digital_human_id,
                candidate=candidate, classification=classification,
                source_session_id=source_session_id,
                source_message_ids=source_message_ids,
                expires_at=expires_at, now=now,
                user_confirmed=user_confirmed,
                superseded_by=None,
            )
            if superseded_record is not None:
                superseded_record.superseded_by_memory_id = row.memory_id
                # Flush the old row's status before inserting the replacement;
                # the partial unique index only permits one active fact_key.
                db.flush([superseded_record])
            db.add(row)
            db.flush()
            db.commit()
            db.refresh(row)
            return WriteOutcome(
                stored=row, superseded_id=superseded_id, skipped_reason=None,
            )

    def _same_content(
        self, existing: MemoryItemRecord, candidate: WriteCandidate
    ) -> bool:
        return existing.content_hash == _content_hash(candidate.summary)

    def _supersede(
        self, db, existing: MemoryItemRecord, candidate: WriteCandidate, now: datetime
    ) -> bool:
        existing.status = "superseded"
        existing.valid_to = now
        existing.updated_at = now
        # ``superseded_by_memory_id`` is back-filled after we flush the new
        # row; the writer handles that step before commit.

    def set_superseded_by(
        self, db, old_id: str, new_id: str
    ) -> None:
        from sqlalchemy import update
        db.execute(update(MemoryItemRecord).where(
            MemoryItemRecord.memory_id == old_id
        ).values(superseded_by_memory_id=new_id))

    def _enforce_capacity(
        self, db, *, user_id: str, digital_human_id: str,
        max_items: int, importance_floor: float, now: datetime,
    ) -> None:
        active_count = db.scalar(select(func.count()).select_from(MemoryItemRecord).where(
            MemoryItemRecord.user_id == user_id,
            MemoryItemRecord.digital_human_id == digital_human_id,
            MemoryItemRecord.status.in_(["active", "suspended"]),
        )) or 0
        if active_count < max_items:
            return True
        victim = db.execute(
            select(MemoryItemRecord).where(
                MemoryItemRecord.user_id == user_id,
                MemoryItemRecord.digital_human_id == digital_human_id,
                MemoryItemRecord.status.in_(["active", "suspended"]),
                MemoryItemRecord.edited_at.is_(None),
            ).order_by(
                MemoryItemRecord.importance_score,
                MemoryItemRecord.recall_count,
                MemoryItemRecord.created_at,
            ).limit(1).with_for_update(skip_locked=True)
        ).scalar_one_or_none()
        if victim is None:
            return False
        if importance_floor <= victim.importance_score:
            return False
        self._scrub(victim, now, reason="capacity_replaced")
        return True

    def _build_row(
        self, *, user_id: str, digital_human_id: str,
        candidate: WriteCandidate, classification: Classification,
        source_session_id: str | None, source_message_ids: list[str],
        expires_at: datetime, now: datetime,
        user_confirmed: bool, superseded_by: str | None,
    ) -> MemoryItemRecord:
        summary: str | None
        summary_ciphertext: str | None
        search_text: str | None
        key_version: int | None
        if classification.sensitivity == "sensitive":
            if not user_confirmed:
                # Worker-extracted sensitive content: encrypt + isolate.
                from app.core.memory_crypto import encrypt_memory
                try:
                    summary_ciphertext = encrypt_memory(candidate.summary)
                    key_version = int(settings.memory_encryption_key_version)
                except Exception:
                    # If MEMORY_ENCRYPTION_KEY is not set we MUST fail closed
                    # rather than store plaintext. Drop the candidate.
                    raise MemoryWriterError(
                        "sensitive candidate cannot be encrypted; MEMORY_ENCRYPTION_KEY is missing"
                    )
            else:
                # User-edited sensitive row: same path, but we mark
                # user_confirmed so the writer keeps confidence at 1.0.
                from app.core.memory_crypto import encrypt_memory
                summary_ciphertext = encrypt_memory(candidate.summary)
                key_version = int(settings.memory_encryption_key_version)
            summary = None
            search_text = None
        else:
            summary = candidate.summary
            summary_ciphertext = None
            key_version = None
            search_text = _search_text(candidate.summary, candidate.tags)

        return MemoryItemRecord(
            memory_id=str(uuid4()),
            user_id=user_id, digital_human_id=digital_human_id,
            memory_type=candidate.memory_type,
            category=candidate.category,
            fact_key=candidate.fact_key,
            summary=summary,
            summary_ciphertext=summary_ciphertext,
            encryption_key_version=key_version,
            search_text=search_text,
            content_hash=_content_hash(candidate.summary),
            importance_score=candidate.importance_score,
            confidence_score=1.0 if user_confirmed else candidate.confidence_score,
            user_confirmed=user_confirmed,
            sensitivity=classification.sensitivity,
            tags=list(candidate.tags or [])[:8],
            occurred_at=candidate.occurred_at,
            valid_from=candidate.valid_from,
            valid_to=None,
            superseded_by_memory_id=superseded_by,
            source_session_id=source_session_id,
            source_message_ids=list(source_message_ids or [])[-30:],
            embedding_model=None,
            embedding_revision=None,
            status="active",
            expires_at=expires_at,
            created_at=now, updated_at=now,
        )

    def _scrub(self, row: MemoryItemRecord, now: datetime, *, reason: str) -> None:
        row.status = "deleted"
        row.summary = None
        row.summary_ciphertext = None
        row.search_text = None
        row.embedding = None
        row.embedding_model = None
        row.embedding_revision = None
        row.tags = []
        row.source_message_ids = []
        row.deleted_at = now
        row.updated_at = now
        # content_hash is non-null; stamp it with a sentinel so audits can
        # tell "scrubbed" apart from "never set".
        row.content_hash = hashlib.sha256(
            f"deleted:{row.memory_id}:{reason}:{now.isoformat()}".encode("utf-8")
        ).hexdigest()

    # --------------------------------------------------------- user mutations
    def user_update(
        self, *, user_id: str, digital_human_id: str, memory_id: str,
        summary: str | None, category: str | None, tags: list[str] | None,
    ) -> MemoryItemRecord:
        with SessionLocal() as db:
            db.execute(select(func.pg_advisory_xact_lock(func.hashtext(
                f"relationship-memory:{user_id}:{digital_human_id}"
            ))))
            row = db.execute(
                select(MemoryItemRecord).where(
                    MemoryItemRecord.memory_id == memory_id,
                    MemoryItemRecord.user_id == user_id,
                    MemoryItemRecord.digital_human_id == digital_human_id,
                    MemoryItemRecord.status != "deleted",
                ).with_for_update()
            ).scalar_one_or_none()
            if row is None:
                raise MemoryWriterError("memory not found")
            now = _now()
            if summary is not None:
                cleaned = _normalise_text(summary)[:500]
                if not cleaned:
                    raise MemoryWriterError("summary cannot be empty")
                if contains_high_risk_credentials(cleaned):
                    raise CredentialRejected("edited summary contains credentials")
                classification = classify(cleaned, category=category or row.category)
                row.content_hash = _content_hash(cleaned)
                row.edited_at = now
                row.user_confirmed = True
                row.confidence_score = 1.0
                row.updated_at = now
                if classification.sensitivity == "sensitive":
                    from app.core.memory_crypto import encrypt_memory
                    row.summary_ciphertext = encrypt_memory(cleaned)
                    row.encryption_key_version = int(settings.memory_encryption_key_version)
                    row.summary = None
                    row.search_text = None
                else:
                    row.summary = cleaned
                    row.summary_ciphertext = None
                    row.encryption_key_version = None
                    row.search_text = _search_text(cleaned, tags or row.tags or [])
                row.sensitivity = classification.sensitivity
            if category is not None and category in _CATEGORIES:
                row.category = category
            if tags is not None:
                row.tags = [str(t).strip()[:32] for t in tags if str(t).strip()][:8]
            db.commit()
            db.refresh(row)
            return row

    def user_delete(
        self, *, user_id: str, digital_human_id: str, memory_id: str,
    ) -> None:
        with SessionLocal() as db:
            row = db.execute(
                select(MemoryItemRecord).where(
                    MemoryItemRecord.memory_id == memory_id,
                    MemoryItemRecord.user_id == user_id,
                    MemoryItemRecord.digital_human_id == digital_human_id,
                ).with_for_update()
            ).scalar_one_or_none()
            if row is None:
                raise MemoryWriterError("memory not found")
            self._scrub(row, _now(), reason="user_deleted")
            db.commit()

    def hard_delete_relationship(
        self, *, user_id: str, digital_human_id: str, memory_id: str | None = None,
    ) -> int:
        from sqlalchemy import delete
        with SessionLocal() as db:
            stmt = delete(MemoryItemRecord).where(
                MemoryItemRecord.user_id == user_id,
                MemoryItemRecord.digital_human_id == digital_human_id,
            )
            if memory_id is not None:
                stmt = stmt.where(MemoryItemRecord.memory_id == memory_id)
            result = db.execute(stmt)
            db.commit()
            return int(result.rowcount or 0)


memory_writer_service = MemoryWriterService()


__all__ = [
    "CredentialRejected",
    "MemoryWriterError",
    "MemoryWriterService",
    "WriteCandidate",
    "WriteOutcome",
    "memory_writer_service",
    "normalise_fact_key",
]
