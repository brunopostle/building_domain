"""Pass 10b — Predicate stabilization.

Two-phase algorithm:
  Phase 1 — embedding-based auto-mapping: compare each non-core predicate
    against CORE_PREDICATES via cosine similarity (all-mpnet-base-v2).
    ≥ 0.85 → auto-map; [0.60, 0.85) → Phase 2; < 0.60 → pending_predicates.
  Phase 2 — LLM disambiguation for predicates in the ambiguous band.
    Uses provider.classify() to pick a core predicate or "none".
    Mapped → update assertions + log to predicate_mappings.
    Unmapped → pending_predicates (dedup by increment occurrence_count).

Mapped predicates are logged to predicate_mappings with reviewer encoding the
method used ("embedding" or "llm:<model_id>").
Completion is tracked in pass_progress with key '10b'.
"""
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import structlog
from sqlmodel import Session, select

from bsos.llm.protocol import LLMProvider
from bsos.persistence.models import (
    AssertionRow,
    PassProgressRow,
    PendingPredicateRow,
    PredicateMappingRow,
)
from bsos.vocab import CORE_PREDICATES

log = structlog.get_logger()

EMBEDDING_MODEL = "all-mpnet-base-v2"
AUTO_MAP_THRESHOLD = 0.85
AMBIGUOUS_LOW = 0.60

CLASSIFY_PROMPT_TEMPLATE = (
    "The predicate '{predicate}' was found in a building-domain knowledge assertion "
    "but is not part of the controlled vocabulary.\n\n"
    "Which of the following core predicates best captures its meaning?\n"
    "{options_list}\n\n"
    "If none of the above is a reasonable semantic match, answer 'none'.\n"
    "Reply with exactly one option from the list."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarities(query_vec: np.ndarray, cand_vecs: np.ndarray) -> np.ndarray:
    query_norm = float(np.linalg.norm(query_vec))
    if query_norm == 0.0:
        return np.zeros(len(cand_vecs))
    cand_norms = np.linalg.norm(cand_vecs, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = (cand_vecs @ query_vec) / (cand_norms * query_norm)
    return np.where(np.isfinite(sims), sims, 0.0)


def _non_core_predicates(session: Session) -> list[str]:
    """Return sorted list of distinct non-core predicates present in assertions."""
    rows = session.exec(select(AssertionRow.predicate).distinct()).all()
    return sorted({p for p in rows if p not in CORE_PREDICATES})


def _already_mapped(session: Session, predicate: str) -> str | None:
    """Return canonical predicate if already recorded in predicate_mappings."""
    row = session.exec(
        select(PredicateMappingRow).where(PredicateMappingRow.from_predicate == predicate)
    ).first()
    return row.to_predicate if row else None


def _update_assertions(session: Session, old_pred: str, new_pred: str) -> int:
    """Rewrite predicate on all assertions using old_pred. Returns count."""
    rows = session.exec(
        select(AssertionRow).where(AssertionRow.predicate == old_pred)
    ).all()
    for row in rows:
        row.predicate = new_pred
    return len(rows)


def _log_mapping(session: Session, from_pred: str, to_pred: str, reviewer: str) -> None:
    session.add(PredicateMappingRow(
        from_predicate=from_pred,
        to_predicate=to_pred,
        created_at=datetime.now(timezone.utc),
        reviewer=reviewer,
    ))


def _upsert_pending(session: Session, predicate: str, now: datetime) -> None:
    """Insert or increment pending_predicates record for predicate."""
    existing = session.exec(
        select(PendingPredicateRow).where(PendingPredicateRow.value == predicate)
    ).first()
    if existing:
        existing.occurrence_count += 1
        existing.last_seen_at = now
    else:
        session.add(PendingPredicateRow(
            value=predicate,
            vocabulary_type="predicate",
            occurrence_count=1,
            first_seen_at=now,
            last_seen_at=now,
            flagged_for_review=False,
        ))


# ---------------------------------------------------------------------------
# Phase 1 — embedding-based auto-mapping
# ---------------------------------------------------------------------------

def _run_phase1(
    engine,
    embedder: Callable[[list[str]], np.ndarray],
    core_list: list[str],
    core_vecs: np.ndarray,
    non_core: list[str],
) -> tuple[dict[str, str], list[str]]:
    """Return (auto_mapped {pred→core}, ambiguous [pred]) after phase 1.

    Also writes pending_predicates for < 0.60 matches and flushes them.
    Auto-mapped predicates are updated in assertions and logged to predicate_mappings.
    """
    auto_mapped: dict[str, str] = {}
    ambiguous: list[str] = []
    now = datetime.now(timezone.utc)

    for pred in non_core:
        with Session(engine) as session:
            canonical = _already_mapped(session, pred)
            if canonical:
                log.debug("pass10b_phase1_skip_already_mapped", predicate=pred, canonical=canonical)
                auto_mapped[pred] = canonical
                continue

            pred_vec = np.array(embedder([pred]), dtype=np.float32)[0]
            sims = _cosine_similarities(pred_vec, core_vecs)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            if best_sim >= AUTO_MAP_THRESHOLD:
                canonical = core_list[best_idx]
                count = _update_assertions(session, pred, canonical)
                _log_mapping(session, pred, canonical, reviewer="embedding")
                session.commit()
                auto_mapped[pred] = canonical
                log.info(
                    "pass10b_phase1_auto_mapped",
                    predicate=pred,
                    canonical=canonical,
                    similarity=round(best_sim, 4),
                    assertions_updated=count,
                )
            elif best_sim >= AMBIGUOUS_LOW:
                ambiguous.append(pred)
                log.debug("pass10b_phase1_ambiguous", predicate=pred, best_sim=round(best_sim, 4))
            else:
                _upsert_pending(session, pred, now)
                session.commit()
                log.info(
                    "pass10b_phase1_pending",
                    predicate=pred,
                    best_sim=round(best_sim, 4),
                )

    return auto_mapped, ambiguous


# ---------------------------------------------------------------------------
# Phase 2 — LLM disambiguation
# ---------------------------------------------------------------------------

def _run_phase2(
    engine,
    provider: LLMProvider,
    core_list: list[str],
    ambiguous: list[str],
) -> None:
    """LLM classification for predicates in the ambiguous band."""
    options = core_list + ["none"]
    options_text = "\n".join(f"- {o}" for o in options)
    now = datetime.now(timezone.utc)

    for pred in ambiguous:
        with Session(engine) as session:
            if _already_mapped(session, pred):
                log.debug("pass10b_phase2_skip_already_mapped", predicate=pred)
                continue

        prompt = CLASSIFY_PROMPT_TEMPLATE.format(
            predicate=pred,
            options_list=options_text,
        )
        try:
            answer = provider.classify(prompt, options)
        except Exception as exc:
            log.warning("pass10b_phase2_llm_error", predicate=pred, error=str(exc))
            with Session(engine) as session:
                _upsert_pending(session, pred, now)
                session.commit()
            continue

        answer = answer.strip().lower()

        with Session(engine) as session:
            if answer in CORE_PREDICATES:
                count = _update_assertions(session, pred, answer)
                _log_mapping(session, pred, answer, reviewer=f"llm:{provider.model_id}")
                session.commit()
                log.info(
                    "pass10b_phase2_mapped",
                    predicate=pred,
                    canonical=answer,
                    assertions_updated=count,
                )
            else:
                _upsert_pending(session, pred, now)
                session.commit()
                log.info("pass10b_phase2_pending", predicate=pred, llm_answer=answer)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pass10b(
    engine,
    embedding_model: str = EMBEDDING_MODEL,
    provider: LLMProvider | None = None,
    _embedder: Callable[[list[str]], np.ndarray] | None = None,
    dry_run: bool = False,
) -> dict:
    """Run Pass 10b: predicate stabilization.

    provider is used for Phase 2 LLM disambiguation. If None, ambiguous
    predicates are sent directly to pending_predicates.
    _embedder is a test seam; omit in production to use SentenceTransformer.
    """
    with Session(engine) as session:
        progress = session.get(PassProgressRow, ("10b", "__global__", embedding_model))
        if progress and progress.status == "completed":
            log.info("pass10b_skip", reason="already completed")
            return {"status": "already_completed"}

    log.info("pass10b_start", embedding_model=embedding_model)

    with Session(engine) as session:
        non_core = _non_core_predicates(session)

    if not non_core:
        log.info("pass10b_nothing_to_do")
    else:
        log.info("pass10b_non_core_found", count=len(non_core), predicates=non_core)

    if dry_run:
        return {
            "dry_run": True,
            "non_core_predicate_count": len(non_core),
            "non_core_predicates": non_core,
        }

    core_list = sorted(CORE_PREDICATES)

    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _st = SentenceTransformer(embedding_model)
        embedder: Callable[[list[str]], np.ndarray] = lambda texts: _st.encode(
            texts, show_progress_bar=False
        )
    else:
        embedder = _embedder

    core_vecs = np.array(embedder(core_list), dtype=np.float32)

    auto_mapped, ambiguous = _run_phase1(engine, embedder, core_list, core_vecs, non_core)

    if ambiguous and provider is not None:
        _run_phase2(engine, provider, core_list, ambiguous)
    elif ambiguous:
        now = datetime.now(timezone.utc)
        log.info(
            "pass10b_phase2_skipped_no_provider",
            ambiguous_count=len(ambiguous),
        )
        for pred in ambiguous:
            with Session(engine) as session:
                _upsert_pending(session, pred, now)
                session.commit()

    with Session(engine) as session:
        now = datetime.now(timezone.utc)
        existing = session.get(PassProgressRow, ("10b", "__global__", embedding_model))
        if existing:
            existing.completed_at = now
            existing.status = "completed"
        else:
            session.add(PassProgressRow(
                pass_number="10b",
                entity_id="__global__",
                model=embedding_model,
                completed_at=now,
                status="completed",
            ))
        session.commit()

    with Session(engine) as session:
        pending_count = len(session.exec(
            select(PendingPredicateRow).where(PendingPredicateRow.vocabulary_type == "predicate")
        ).all())
        mapped_count = len(auto_mapped)
        phase2_count = len([p for p in ambiguous]) if ambiguous else 0

    result = {
        "auto_mapped": mapped_count,
        "phase2_processed": phase2_count,
        "pending_predicates": pending_count,
    }
    log.info("pass10b_complete", **result)
    return result
