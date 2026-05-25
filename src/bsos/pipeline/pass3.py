"""Pass 3 — Relationship Extraction with cross-prompt consistency.

For each non-merged entity: runs N distinct prompt framings to extract Assertion
records, groups semantically equivalent assertions across framings using LAP
matching (scipy), computes mean pairwise cosine similarity as
cross_prompt_consistency, and writes canonical AssertionRow records.

Implements pass_progress resume semantics (entity-level) and 4-worker
ThreadPoolExecutor for entity-level parallelism.
"""
import itertools
import json
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from statistics import mean as stat_mean
from typing import Callable

import numpy as np
import structlog
from scipy.optimize import linear_sum_assignment
from sqlmodel import Session, select

from bsos.llm.protocol import LLMProvider
from bsos.persistence.models import (
    AssertionRow, EntityAliasRow, EntityRow, PassProgressRow,
)
from bsos.pipeline.schemas import AssertionExtractionResponse, ExtractedAssertion

log = structlog.get_logger()

CONSISTENCY_THRESHOLD = 0.70
EMBEDDING_MODEL = "all-mpnet-base-v2"

FRAMING_TEMPLATES = [
    (
        "For the building concept '{name}' (type: {entity_type}), list all building components, "
        "systems, materials, or activities that '{name}' REQUIRES or DEPENDS ON to function "
        "properly, and anything it SUPPORTS, PROTECTS FROM damage, or CONTAINS. "
        "For each relationship specify: the predicate "
        "(requires/depends_on/supports/protects_from/contains/connects_to/improves/"
        "conflicts_with/unsuitable_for), the related concept name, any conditions, "
        "exceptions, applicable building types, confidence, and your reasoning."
    ),
    (
        "Describe the outgoing relationships of '{name}' (type: {entity_type}) "
        "from a construction-sequence and systems-integration perspective. "
        "What must '{name}' CONNECT TO, what does it IMPROVE, what does it CONFLICT WITH "
        "or is UNSUITABLE FOR, and what does it ultimately DEPEND ON? "
        "For each: predicate, related concept, conditions, exceptions, applicability, "
        "confidence (0–1), and rationale."
    ),
    (
        "From a building-physics and regulatory standpoint, what relationships does "
        "'{name}' (type: {entity_type}) have with other building concepts? "
        "Consider structural, thermal, fire, acoustic, and legal dimensions. "
        "List each relationship with predicate, object concept name, conditions under which "
        "it applies, exceptions, applicable building types, confidence, and reasoning."
    ),
]


# ---------------------------------------------------------------------------
# Cross-prompt consistency helpers
# ---------------------------------------------------------------------------

def _assertion_text(entity_name: str, a: ExtractedAssertion) -> str:
    return f"{entity_name} {a.predicate} {a.object_name}"


def _group_assertions(
    framing_lists: list[list[ExtractedAssertion]],
    entity_name: str,
    embedder: Callable[[list[str]], np.ndarray],
    threshold: float = CONSISTENCY_THRESHOLD,
) -> list[dict]:
    """Group assertions across framings by semantic similarity via LAP matching.

    Returns list of dicts:
      assertions: list[ExtractedAssertion]  — one per matched framing
      consistency: float | None             — mean pairwise cosine sim (None if singleton)
      framing_count: int
    """
    # Flatten all (framing_idx, local_idx, assertion)
    tagged: list[tuple[int, int, ExtractedAssertion]] = []
    for fi, assertions in enumerate(framing_lists):
        for ai, a in enumerate(assertions):
            tagged.append((fi, ai, a))

    if not tagged:
        return []

    texts = [_assertion_text(entity_name, a) for _, _, a in tagged]
    raw_vecs = np.array(embedder(texts), dtype=np.float32)
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    vecs = raw_vecs / np.maximum(norms, 1e-10)

    # Union-Find over flat indices into `tagged`
    parent = list(range(len(tagged)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    n_framings = len(framing_lists)
    for fi, fj in itertools.combinations(range(n_framings), 2):
        i_indices = [k for k, (f, _, _) in enumerate(tagged) if f == fi]
        j_indices = [k for k, (f, _, _) in enumerate(tagged) if f == fj]
        if not i_indices or not j_indices:
            continue

        sim = np.array([
            [float(np.dot(vecs[i], vecs[j])) for j in j_indices]
            for i in i_indices
        ])
        row_ind, col_ind = linear_sum_assignment(-sim)
        for ri, ci in zip(row_ind, col_ind):
            if sim[ri, ci] >= threshold:
                union(i_indices[ri], j_indices[ci])

    # Collect groups
    groups_dict: dict[int, list[int]] = defaultdict(list)
    for k in range(len(tagged)):
        groups_dict[find(k)].append(k)

    result = []
    for members in groups_dict.values():
        assertions = [tagged[k][2] for k in members]
        if len(members) == 1:
            result.append({"assertions": assertions, "consistency": None, "framing_count": 1})
        else:
            pairs = list(itertools.combinations(members, 2))
            sims = [float(np.dot(vecs[a], vecs[b])) for a, b in pairs]
            result.append({
                "assertions": assertions,
                "consistency": float(np.mean(sims)),
                "framing_count": len(members),
            })

    return result


# ---------------------------------------------------------------------------
# Per-entity worker
# ---------------------------------------------------------------------------

def _build_name_lookup(engine) -> dict[str, tuple[str, str]]:
    """Return lowercase_name → (entity_id, entity_type) for all active entities and aliases."""
    lookup: dict[str, tuple[str, str]] = {}
    with Session(engine) as s:
        for row in s.exec(select(EntityRow).where(EntityRow.status != "merged")).all():
            lookup[row.name.lower()] = (row.id, row.entity_type)
        for alias_row in s.exec(select(EntityAliasRow)).all():
            entity = s.get(EntityRow, alias_row.entity_id)
            if entity and entity.status != "merged":
                lookup[alias_row.alias.lower()] = (entity.id, entity.entity_type)
    return lookup


def _process_entity(
    engine,
    entity_id: str,
    entity_name: str,
    entity_type: str,
    provider: LLMProvider,
    name_lookup: dict[str, tuple[str, str]],
    embedder: Callable[[list[str]], np.ndarray],
    n_framings: int,
    run_id: str,
) -> int:
    """Extract and write assertions for one entity. Returns count of assertions written."""
    with Session(engine) as session:
        progress = session.get(PassProgressRow, ("3", entity_id, provider.model_id))
        if progress and progress.status == "completed":
            log.debug("pass3_entity_skip_resume", entity=entity_name)
            return 0

        framing_lists: list[list[ExtractedAssertion]] = []
        successful_framings = 0
        for template in FRAMING_TEMPLATES[:n_framings]:
            prompt = template.format(name=entity_name, entity_type=entity_type)
            try:
                response = provider.extract(
                    prompt, AssertionExtractionResponse, entity_name=entity_name
                )
                framing_lists.append(response.assertions)
                successful_framings += 1
            except Exception as exc:
                log.warning("pass3_extraction_failed", entity=entity_name, error=str(exc))
                framing_lists.append([])

        if successful_framings == 0:
            log.warning("pass3_entity_all_framings_failed", entity=entity_name)
            return 0

        groups = _group_assertions(framing_lists, entity_name, embedder)

        now = datetime.now(timezone.utc)
        written = 0

        for group in groups:
            canonical: ExtractedAssertion = group["assertions"][0]

            if not canonical.object_name.strip():
                continue

            obj_key = canonical.object_name.lower().strip()
            obj_info = name_lookup.get(obj_key)
            if obj_info is None:
                log.debug("pass3_unresolved_object", entity=entity_name, object=canonical.object_name)
                continue

            object_id, object_type = obj_info
            if object_id == entity_id:
                continue

            mean_confidence = stat_mean(a.confidence for a in group["assertions"])

            row = AssertionRow(
                id=str(uuid.uuid4()),
                subject_id=entity_id,
                predicate=canonical.predicate,
                object_id=object_id,
                subject_type=entity_type,
                object_type=object_type,
                conditions=json.dumps(canonical.conditions),
                exceptions=json.dumps(canonical.exceptions),
                applicability=json.dumps(canonical.applicability),
                confidence=round(mean_confidence, 4),
                status="proposed",
                knowledge_origin=canonical.knowledge_origin,
                rationale=canonical.rationale,
                cross_prompt_consistency=group["consistency"],
                prompt_framing_count=group["framing_count"],
                source_model=provider.model_id,
                source_prompt=FRAMING_TEMPLATES[0].format(name=entity_name, entity_type=entity_type),
                created_at=now,
                extraction_run_id=run_id,
            )
            session.add(row)
            written += 1

        existing_progress = session.get(PassProgressRow, ("3", entity_id, provider.model_id))
        if existing_progress:
            existing_progress.completed_at = now
            existing_progress.status = "completed"
        else:
            session.add(PassProgressRow(
                pass_number="3",
                entity_id=entity_id,
                model=provider.model_id,
                completed_at=now,
                status="completed",
            ))

        session.commit()
        log.info("pass3_entity_done", entity=entity_name, assertions_written=written)
        return written


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pass3(
    engine,
    provider: LLMProvider,
    run_id: str,
    _embedder: Callable[[list[str]], np.ndarray] | None = None,
    n_framings: int = 3,
    dry_run: bool = False,
    max_workers: int = 4,
) -> dict:
    """Run Pass 3: relationship extraction for all active entities.

    engine: SQLAlchemy Engine (not session) — workers create their own sessions.
    _embedder: test seam; omit in production to use SentenceTransformer.

    Returns summary dict: {entities_processed, assertions_written}.
    """
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(EMBEDDING_MODEL)
        embedder: Callable[[list[str]], np.ndarray] = lambda texts: st_model.encode(
            texts, show_progress_bar=False
        )
    else:
        embedder = _embedder

    name_lookup = _build_name_lookup(engine)

    with Session(engine) as session:
        entities = session.exec(
            select(EntityRow).where(EntityRow.status != "merged")
        ).all()
        entity_tuples = [(e.id, e.name, e.entity_type) for e in entities]

    log.info("pass3_start", entity_count=len(entity_tuples), n_framings=n_framings)

    if dry_run:
        log.info("pass3_dry_run", entities=len(entity_tuples))
        return {"entities_processed": len(entity_tuples), "assertions_written": 0}

    total_assertions = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _process_entity,
                engine, eid, ename, etype,
                provider, name_lookup, embedder, n_framings, run_id,
            ): ename
            for eid, ename, etype in entity_tuples
        }
        for future in as_completed(futures):
            entity_name = futures[future]
            try:
                total_assertions += future.result()
            except Exception as exc:
                log.error("pass3_entity_error", entity=entity_name, error=str(exc))

    log.info("pass3_complete", entities_processed=len(entity_tuples), assertions_written=total_assertions)
    return {"entities_processed": len(entity_tuples), "assertions_written": total_assertions}
