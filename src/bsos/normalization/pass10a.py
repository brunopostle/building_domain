"""Pass 10a — ref resolution.

Resolves three types of free-text references left by Passes 8-9:
  1. Pattern.force_descriptions → Force records (exact name, then cosine sim ≥ 0.85)
  2. Pattern.related_pattern_names → Pattern records (same two-step match)
  3. Retry pending_entity_refs against Entity table (exact match only)

Per-pattern atomicity: each pattern is committed in a separate transaction.
Resumable: patterns with empty force_descriptions/related_pattern_names are skipped.
"""
import json
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import structlog
from sqlmodel import Session, select

from bsos.config import set_config
from bsos.persistence.models import (
    EntityAliasRow,
    EntityRow,
    ForceRow,
    PassProgressRow,
    PatternRow,
    PendingEntityRefRow,
    PendingForceRefRow,
    PendingPatternRefRow,
)

log = structlog.get_logger()

EMBEDDING_MODEL = "all-mpnet-base-v2"
SIMILARITY_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _cosine_similarities(query_vec: np.ndarray, cand_vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and a matrix of candidate vectors."""
    query_norm = float(np.linalg.norm(query_vec))
    if query_norm == 0.0:
        return np.zeros(len(cand_vecs))
    cand_norms = np.linalg.norm(cand_vecs, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = (cand_vecs @ query_vec) / (cand_norms * query_norm)
    sims = np.where(np.isfinite(sims), sims, 0.0)
    return sims


def _exact_match_index(query: str, candidates: list[str]) -> int | None:
    """Case-insensitive exact match; returns index or None."""
    q = query.lower().strip()
    for i, c in enumerate(candidates):
        if c.lower().strip() == q:
            return i
    return None


def _find_match(
    query: str,
    candidates: list[str],
    cand_vecs: np.ndarray,
    embedder: Callable[[list[str]], np.ndarray],
    threshold: float = SIMILARITY_THRESHOLD,
) -> int | None:
    """Exact match first; then cosine similarity against pre-embedded candidates."""
    idx = _exact_match_index(query, candidates)
    if idx is not None:
        return idx
    if len(candidates) == 0:
        return None
    query_vec = np.array(embedder([query]), dtype=np.float32)[0]
    sims = _cosine_similarities(query_vec, cand_vecs)
    best = int(np.argmax(sims))
    return best if float(sims[best]) >= threshold else None


# ---------------------------------------------------------------------------
# Lookup builders
# ---------------------------------------------------------------------------

def _load_forces(session: Session) -> tuple[list[str], list[str]]:
    """Return (ids, names) for all non-deprecated Forces."""
    rows = session.exec(
        select(ForceRow).where(ForceRow.status != "deprecated")
    ).all()
    return [r.id for r in rows], [r.name for r in rows]


def _load_patterns(session: Session) -> tuple[list[str], list[str]]:
    """Return (ids, names) for all non-deprecated Patterns."""
    rows = session.exec(
        select(PatternRow).where(PatternRow.status != "deprecated")
    ).all()
    return [r.id for r in rows], [r.name for r in rows]


def _entity_lookup(session: Session) -> dict[str, str]:
    """Return lowercase name/alias → entity_id for all active entities."""
    lookup: dict[str, str] = {}
    for row in session.exec(
        select(EntityRow).where(EntityRow.status != "merged")
    ).all():
        lookup[row.name.lower()] = row.id
    for alias in session.exec(select(EntityAliasRow)).all():
        entity = session.get(EntityRow, alias.entity_id)
        if entity and entity.status != "merged":
            lookup[alias.alias.lower()] = entity.id
    return lookup


# ---------------------------------------------------------------------------
# Resolution steps
# ---------------------------------------------------------------------------

def _resolve_force_descriptions(
    engine,
    embedder: Callable[[list[str]], np.ndarray],
) -> dict:
    """Resolve Pattern.force_descriptions → Pattern.force_ids."""
    with Session(engine) as session:
        f_ids, f_names = _load_forces(session)
        pattern_ids = [
            r.id for r in session.exec(select(PatternRow)).all()
        ]

    if not f_names:
        log.info("pass10a_force_descriptions_skip", reason="no forces in database")
        return {"resolved": 0, "unresolved": 0}

    # Pre-embed all force names once.
    f_vecs = np.array(embedder(f_names), dtype=np.float32)

    resolved_total = 0
    unresolved_total = 0

    for pattern_id in pattern_ids:
        with Session(engine) as session:
            pattern = session.get(PatternRow, pattern_id)
            if pattern is None:
                continue

            descriptions = json.loads(pattern.force_descriptions or "[]")
            if not descriptions:
                continue  # Already resolved — resumable skip

            existing_ids: list[str] = json.loads(pattern.force_ids or "[]")
            now = datetime.now(timezone.utc)

            for desc in descriptions:
                desc = desc.strip()
                if not desc:
                    continue
                match_idx = _find_match(desc, f_names, f_vecs, embedder)
                if match_idx is not None:
                    matched_id = f_ids[match_idx]
                    if matched_id not in existing_ids:
                        existing_ids.append(matched_id)
                    resolved_total += 1
                    log.debug(
                        "pass10a_force_desc_resolved",
                        desc=desc,
                        matched_force=f_names[match_idx],
                    )
                else:
                    session.add(PendingForceRefRow(
                        description=desc,
                        failure_type="unresolved_ref",
                        pattern_id=pattern_id,
                        created_at=now,
                    ))
                    unresolved_total += 1
                    log.debug("pass10a_force_desc_unresolved", desc=desc, pattern_id=pattern_id)

            pattern.force_ids = json.dumps(existing_ids)
            pattern.force_descriptions = json.dumps([])
            session.commit()

    log.info(
        "pass10a_force_descriptions_done",
        resolved=resolved_total,
        unresolved=unresolved_total,
    )
    return {"resolved": resolved_total, "unresolved": unresolved_total}


def _resolve_related_pattern_names(
    engine,
    embedder: Callable[[list[str]], np.ndarray],
) -> dict:
    """Resolve Pattern.related_pattern_names → Pattern.related_pattern_ids."""
    with Session(engine) as session:
        all_p_ids, all_p_names = _load_patterns(session)
        pattern_ids = [r.id for r in session.exec(select(PatternRow)).all()]

    if not all_p_names:
        return {"resolved": 0, "unresolved": 0}

    # Pre-embed all pattern names once.
    p_vecs = np.array(embedder(all_p_names), dtype=np.float32)

    resolved_total = 0
    unresolved_total = 0

    for pattern_id in pattern_ids:
        with Session(engine) as session:
            pattern = session.get(PatternRow, pattern_id)
            if pattern is None:
                continue

            related_names = json.loads(pattern.related_pattern_names or "[]")
            if not related_names:
                continue  # Already resolved — resumable skip

            existing_related: list[str] = json.loads(pattern.related_pattern_ids or "[]")
            now = datetime.now(timezone.utc)

            # Exclude self from candidate set.
            self_pos = all_p_ids.index(pattern_id) if pattern_id in all_p_ids else None
            p_ids_filtered = [i for j, i in enumerate(all_p_ids) if j != self_pos]
            p_names_filtered = [n for j, n in enumerate(all_p_names) if j != self_pos]
            p_vecs_filtered = np.delete(p_vecs, self_pos, axis=0) if self_pos is not None else p_vecs

            for name in related_names:
                name = name.strip()
                if not name:
                    continue
                match_idx = _find_match(name, p_names_filtered, p_vecs_filtered, embedder)
                if match_idx is not None:
                    matched_id = p_ids_filtered[match_idx]
                    if matched_id not in existing_related:
                        existing_related.append(matched_id)
                    resolved_total += 1
                    log.debug(
                        "pass10a_related_resolved",
                        name=name,
                        matched=p_names_filtered[match_idx],
                    )
                else:
                    session.add(PendingPatternRefRow(
                        pattern_name=name,
                        source_pattern_id=pattern_id,
                        created_at=now,
                    ))
                    unresolved_total += 1
                    log.debug("pass10a_related_unresolved", name=name, pattern_id=pattern_id)

            pattern.related_pattern_ids = json.dumps(existing_related)
            pattern.related_pattern_names = json.dumps([])
            session.commit()

    log.info(
        "pass10a_related_pattern_names_done",
        resolved=resolved_total,
        unresolved=unresolved_total,
    )
    return {"resolved": resolved_total, "unresolved": unresolved_total}


def _resolve_entity_refs(engine) -> dict:
    """Retry pending_entity_refs against Entity table (exact match only)."""
    with Session(engine) as session:
        entity_lkp = _entity_lookup(session)
        refs = [
            (r.id, r.entity_name, r.source_force_id)
            for r in session.exec(select(PendingEntityRefRow)).all()
        ]

    resolved = 0
    remaining = 0

    for ref_id, entity_name, source_force_id in refs:
        entity_uuid = entity_lkp.get(entity_name.lower().strip())
        if entity_uuid is None:
            remaining += 1
            continue

        with Session(engine) as session:
            force = session.get(ForceRow, source_force_id)
            if force is not None:
                current_affects: list[str] = json.loads(force.affects or "[]")
                if entity_uuid not in current_affects:
                    current_affects.append(entity_uuid)
                    force.affects = json.dumps(current_affects)

            ref_row = session.get(PendingEntityRefRow, ref_id)
            if ref_row is not None:
                session.delete(ref_row)

            session.commit()
        resolved += 1
        log.debug("pass10a_entity_ref_resolved", entity=entity_name, force_id=source_force_id)

    log.info("pass10a_entity_refs_done", resolved=resolved, remaining=remaining)
    return {"resolved": resolved, "remaining": remaining}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pass10a(
    engine,
    embedding_model: str = EMBEDDING_MODEL,
    _embedder: Callable[[list[str]], np.ndarray] | None = None,
    dry_run: bool = False,
) -> dict:
    """Run Pass 10a: resolve all free-text refs left by Passes 8-9.

    Returns summary dict with resolution counts.
    _embedder is a test seam; omit in production to use SentenceTransformer.
    """
    with Session(engine) as session:
        progress = session.get(PassProgressRow, ("10a", "__global__", embedding_model))
        if progress and progress.status == "completed":
            log.info("pass10a_skip", reason="already completed")
            return {"status": "already_completed"}

    log.info("pass10a_start", embedding_model=embedding_model)

    if dry_run:
        with Session(engine) as session:
            patterns_fd = [
                r for r in session.exec(select(PatternRow)).all()
                if json.loads(r.force_descriptions or "[]")
            ]
            patterns_rn = [
                r for r in session.exec(select(PatternRow)).all()
                if json.loads(r.related_pattern_names or "[]")
            ]
            pending_ent = session.exec(select(PendingEntityRefRow)).all()
        return {
            "dry_run": True,
            "patterns_with_force_descriptions": len(patterns_fd),
            "patterns_with_related_names": len(patterns_rn),
            "pending_entity_refs": len(list(pending_ent)),
        }

    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _st = SentenceTransformer(embedding_model)
        embedder: Callable[[list[str]], np.ndarray] = lambda texts: _st.encode(
            texts, show_progress_bar=False
        )
    else:
        embedder = _embedder

    force_result = _resolve_force_descriptions(engine, embedder)
    pattern_result = _resolve_related_pattern_names(engine, embedder)
    entity_result = _resolve_entity_refs(engine)

    with Session(engine) as session:
        now = datetime.now(timezone.utc)
        existing = session.get(PassProgressRow, ("10a", "__global__", embedding_model))
        if existing:
            existing.completed_at = now
            existing.status = "completed"
        else:
            session.add(PassProgressRow(
                pass_number="10a",
                entity_id="__global__",
                model=embedding_model,
                completed_at=now,
                status="completed",
            ))
        set_config(session, "passes_3_9_refs_resolved", "1")
        session.commit()

    result = {
        "force_descriptions_resolved": force_result["resolved"],
        "force_descriptions_unresolved": force_result["unresolved"],
        "pattern_names_resolved": pattern_result["resolved"],
        "pattern_names_unresolved": pattern_result["unresolved"],
        "entity_refs_resolved": entity_result["resolved"],
        "entity_refs_remaining": entity_result["remaining"],
    }
    log.info("pass10a_complete", **result)
    return result
