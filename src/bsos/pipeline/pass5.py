"""Pass 5 — Process/Sequence Extraction.

For each active activity or component entity, ask the LLM what must happen before
and after it in construction sequencing. Extracts ProcessRelation records. Unknown
activity names are created inline as proposed EntityRow(activity). Duplicate
(predecessor_id, successor_id, source_model) triplets are silently skipped;
hard_constraint divergence from an existing row triggers a structlog ERROR.

Materials, spaces, systems, and ifc_class entities are skipped — construction
sequencing constraints apply to activities and physical components, not to
materials, spaces, systems, or IFC schema classes.
"""
import re
import threading
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import structlog
from sqlmodel import Session, func, select

from bsos.llm.protocol import LLMProvider
from bsos.persistence.merge import merge_entity
from bsos.persistence.models import (
    AssertionRow, EntityAliasRow, EntityRow, PassProgressRow, ProcessRelationRow,
)
from bsos.persistence.retry import with_db_retry
from bsos.pipeline.schemas import ProcessRelationExtractionResponse

log = structlog.get_logger()

# Embedding-clustering threshold for the activity-dedup step (run_activity_dedup).
# Calibrated against the production activity set (building_domain-e9k): 0.04 (Pass
# 2's value) is too timid for activity wording variants — it leaves "Install Roof
# Decking" distinct from "Roof Decking Installation". 0.12+ starts collapsing
# genuinely-distinct construction phases (preparation vs installation vs
# completion). 0.08 folds word-order / punctuation / "Install X" ≈ "X Installation"
# / decking ≈ sheathing variants while keeping distinct phases apart.
ACTIVITY_DEDUP_THRESHOLD = 0.08

PROMPT_TEMPLATE = (
    "For the building activity or entity '{name}' (type: {entity_type}), describe its "
    "temporal ordering constraints in a construction sequence. "
    "What activities or processes must be completed BEFORE '{name}' can start? "
    "What activities or processes can only begin AFTER '{name}' is complete? "
    "For each ordering relationship specify: the predecessor activity name, the successor "
    "activity name, whether it is a hard physical constraint (true) or a soft preference "
    "(false), and a clear rationale for why this ordering is required."
)

_creation_lock = threading.Lock()


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


# PROMPT_TEMPLATE always opens with the subject entity's name and type, so the
# subject can be recovered from source_prompt for rows written before subject_id
# existed (building_domain-eue backfill).
_SUBJECT_FROM_PROMPT_RE = re.compile(
    r"^For the building activity or entity '(.+?)' \(type: \w+\)"
)


def backfill_subject_id(engine) -> dict:
    """Populate subject_id for existing process_relations rows from source_prompt.

    Rows written before subject_id existed only recorded the subject entity's
    name inside the unstructured source_prompt text. Rows whose source_prompt
    is missing, doesn't match the expected prefix, or names an entity that no
    longer resolves are left with subject_id=NULL — the safe "treat as
    universal" default cycle detection already applies, not an error.
    """
    name_lookup = _build_name_lookup(engine)
    updated = 0
    skipped = 0

    with Session(engine) as session:
        rows = session.exec(
            select(ProcessRelationRow).where(ProcessRelationRow.subject_id.is_(None))
        ).all()
        total = len(rows)

        for row in rows:
            match = _SUBJECT_FROM_PROMPT_RE.match(row.source_prompt or "")
            if not match:
                skipped += 1
                continue
            resolved = name_lookup.get(match.group(1).lower())
            if resolved is None:
                skipped += 1
                continue
            row.subject_id = resolved[0]
            updated += 1

        session.commit()

    log.info("pass5_backfill_subject_id", updated=updated, skipped=skipped, total=total)
    return {"updated": updated, "skipped": skipped, "total": total}


def _get_or_create_activity(
    session: Session,
    name: str,
    source_model: str,
    run_id: str,
    creating_entity_name: str,
) -> str:
    """Case-insensitive lookup; creates EntityRow(activity) if not found. Returns entity_id."""
    existing = session.exec(
        select(EntityRow)
        .where(EntityRow.name.ilike(name))  # type: ignore[attr-defined]
        .where(EntityRow.status != "merged")
    ).first()
    if existing:
        return existing.id

    new_id = str(uuid.uuid4())
    session.add(EntityRow(
        id=new_id,
        name=name,
        entity_type="activity",
        status="proposed",
        source_model=source_model,
        created_at=datetime.now(timezone.utc),
        extraction_run_id=run_id,
    ))
    session.flush()
    log.warning(
        "pass5_inline_activity_created",
        activity=name,
        creating_entity=creating_entity_name,
    )
    return new_id


def _process_entity(
    engine,
    entity_id: str,
    entity_name: str,
    entity_type: str,
    provider: LLMProvider,
    run_id: str,
) -> tuple[int, list[dict]]:
    """Extract and write process relations for one entity.

    Returns (relations_written, divergences) where divergences is a list of
    dicts describing hard_constraint conflicts with existing rows.
    """
    with Session(engine) as session:
        progress = session.get(PassProgressRow, ("5", entity_id, provider.model_id))
        if progress and progress.status == "completed":
            log.debug("pass5_entity_skip_resume", entity=entity_name)
            return 0, []

        prompt = PROMPT_TEMPLATE.format(name=entity_name, entity_type=entity_type)
        try:
            response = provider.extract(
                prompt, ProcessRelationExtractionResponse, entity_name=entity_name
            )
        except Exception as exc:
            log.warning("pass5_extraction_failed", entity=entity_name, error=str(exc))
            return 0, []

        now = datetime.now(timezone.utc)
        written = 0
        divergences: list[dict] = []

        for extracted in response.process_relations:
            if not extracted.rationale or not extracted.rationale.strip():
                log.debug("pass5_skip_empty_rationale",
                          predecessor=extracted.predecessor_name,
                          successor=extracted.successor_name)
                continue

            if not extracted.predecessor_name.strip() or not extracted.successor_name.strip():
                continue

            with _creation_lock:
                pred_id = _get_or_create_activity(
                    session, extracted.predecessor_name,
                    provider.model_id, run_id, entity_name,
                )
                succ_id = _get_or_create_activity(
                    session, extracted.successor_name,
                    provider.model_id, run_id, entity_name,
                )

            if pred_id == succ_id:
                continue

            # Deduplication: check for existing row with same (pred, succ, model)
            existing = session.exec(
                select(ProcessRelationRow).where(
                    ProcessRelationRow.predecessor_id == pred_id,
                    ProcessRelationRow.successor_id == succ_id,
                    ProcessRelationRow.source_model == provider.model_id,
                )
            ).first()

            if existing:
                if existing.hard_constraint != extracted.hard_constraint:
                    log.error(
                        "pass5_hard_constraint_divergence",
                        predecessor=extracted.predecessor_name,
                        successor=extracted.successor_name,
                        existing=existing.hard_constraint,
                        new=extracted.hard_constraint,
                    )
                    divergences.append({
                        "predecessor": extracted.predecessor_name,
                        "successor": extracted.successor_name,
                        "existing_hard": existing.hard_constraint,
                        "new_hard": extracted.hard_constraint,
                    })
                continue

            session.add(ProcessRelationRow(
                id=str(uuid.uuid4()),
                predecessor_id=pred_id,
                successor_id=succ_id,
                hard_constraint=extracted.hard_constraint,
                source_model=provider.model_id,
                source_prompt=prompt,
                created_at=now,
                extraction_run_id=run_id,
                confidence=0.8,
                status="proposed",
                knowledge_origin="engineering",
                rationale=extracted.rationale.strip(),
                subject_id=entity_id,
            ))
            written += 1

        existing_progress = session.get(PassProgressRow, ("5", entity_id, provider.model_id))
        if existing_progress:
            existing_progress.completed_at = now
            existing_progress.status = "completed"
        else:
            session.add(PassProgressRow(
                pass_number="5",
                entity_id=entity_id,
                model=provider.model_id,
                completed_at=now,
                status="completed",
            ))

        session.commit()
        log.info("pass5_entity_done", entity=entity_name, relations_written=written)
        return written, divergences


ENTITY_TYPES = frozenset({"activity", "component"})


def run_pass5(
    engine,
    provider: LLMProvider,
    run_id: str,
    dry_run: bool = False,
    max_workers: int = 4,
) -> dict:
    """Run Pass 5: process/sequence extraction for activity and component entities.

    Returns summary dict: {entities_processed, relations_written, hard_constraint_divergences}.
    """
    with Session(engine) as session:
        entities = session.exec(
            select(EntityRow).where(
                EntityRow.status != "merged",
                EntityRow.entity_type.in_(ENTITY_TYPES),  # type: ignore[attr-defined]
            )
        ).all()
        entity_tuples = [(e.id, e.name, e.entity_type) for e in entities]

    log.info("pass5_start", entity_count=len(entity_tuples))

    if dry_run:
        log.info("pass5_dry_run", entities=len(entity_tuples))
        return {"entities_processed": len(entity_tuples), "relations_written": 0,
                "hard_constraint_divergences": 0}

    total_written = 0
    all_divergences: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                with_db_retry,
                _process_entity,
                engine, eid, ename, etype, provider, run_id,
            ): ename
            for eid, ename, etype in entity_tuples
        }
        for future in as_completed(futures):
            entity_name = futures[future]
            try:
                written, divergences = future.result()
                total_written += written
                all_divergences.extend(divergences)
            except Exception as exc:
                log.error("pass5_entity_error", entity=entity_name, error=str(exc))

    log.info(
        "pass5_complete",
        entities_processed=len(entity_tuples),
        relations_written=total_written,
        divergences=len(all_divergences),
    )
    return {
        "entities_processed": len(entity_tuples),
        "relations_written": total_written,
        "hard_constraint_divergences": len(all_divergences),
    }


def _activity_degree(session: Session, entity_id: str) -> tuple[int, int]:
    """Return (process_relation_degree, assertion_count) for canonical election.

    Pass 5 mints most duplicate activities inline with zero assertions, so the
    process-relation degree (how many sequencing edges touch the entity) is the
    more meaningful signal of which wording the graph actually settled on.
    """
    proc = session.exec(
        select(func.count(ProcessRelationRow.id)).where(
            (ProcessRelationRow.predecessor_id == entity_id)
            | (ProcessRelationRow.successor_id == entity_id)
        )
    ).one()
    asrt = session.exec(
        select(func.count(AssertionRow.id)).where(
            (AssertionRow.subject_id == entity_id) | (AssertionRow.object_id == entity_id)
        )
    ).one()
    return proc, asrt


def _elect_activity_canonical(session: Session, members: list[EntityRow]) -> str:
    """Pick the canonical activity: most process edges, then most assertions,
    then oldest, then lexicographically-smallest name (deterministic tiebreak)."""
    def sort_key(e: EntityRow) -> tuple:
        proc, asrt = _activity_degree(session, e.id)
        # Sorted ascending; the element that sorts LAST is canonical, so we want
        # high proc/asrt, old created_at, and small name to land at the end.
        return (proc, asrt, -e.created_at.timestamp(), [-ord(c) for c in e.name])

    return max(members, key=sort_key).id


def run_activity_dedup(
    session: Session,
    run_id: str,
    embedding_model: str = "all-mpnet-base-v2",
    distance_threshold: float = ACTIVITY_DEDUP_THRESHOLD,
    _embedder: Callable[[list[str]], np.ndarray] | None = None,
    dry_run: bool = False,
) -> dict:
    """Fold near-duplicate ``activity`` entities minted by Pass 5 into canonicals.

    Pass 5's ``_get_or_create_activity`` matches existing activities by exact
    (case-insensitive) name only, so every wording variant the LLM returns
    ('Roof Sheathing Installation', 'Install Roof Sheathing', 'Roof Decking /
    Sheathing', ...) becomes a distinct entity (building_domain-e9k). This step
    runs after Pass 5: it embeds every active activity name, clusters with the
    same Agglomerative(cosine, average) machinery as Pass 2, and merges each
    cluster into one canonical via :func:`merge_entity` — which repoints the
    ``process_relations`` FKs so the sequencing graph is not stranded.

    Scoped to ``entity_type='activity'`` so it never touches components, spaces,
    systems, materials or seeded ifc_class rows. ``_embedder`` is a test seam.
    """
    from sklearn.cluster import AgglomerativeClustering

    from bsos.pipeline.pass2 import _load_or_compute_embeddings

    activities = session.exec(
        select(EntityRow).where(
            EntityRow.status != "merged",
            EntityRow.entity_type == "activity",
        )
    ).all()

    log.info("activity_dedup_start", activity_count=len(activities),
             threshold=distance_threshold)

    if len(activities) < 2:
        return {"clusters_found": 0, "entities_merged": 0,
                "activities_before": len(activities)}

    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        model_obj = SentenceTransformer(embedding_model)
        embedder: Callable[[list[str]], np.ndarray] = lambda texts: model_obj.encode(
            texts, show_progress_bar=False
        )
    else:
        embedder = _embedder

    vectors = _load_or_compute_embeddings(session, activities, embedding_model, embedder)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    labels = clustering.fit_predict(vectors)

    clusters: dict[int, list[EntityRow]] = defaultdict(list)
    for entity, label in zip(activities, labels):
        clusters[label].append(entity)

    merge_clusters = [m for m in clusters.values() if len(m) >= 2]
    log.info("activity_dedup_clusters", total_clusters=len(clusters),
             merge_clusters=len(merge_clusters))

    entities_merged = 0
    for members in merge_clusters:
        canonical_id = _elect_activity_canonical(session, members)
        canonical_name = next(m.name for m in members if m.id == canonical_id)
        dup_names = [m.name for m in members if m.id != canonical_id]
        log.info("activity_dedup_merge", canonical=canonical_name, duplicates=dup_names)
        entities_merged += len(members) - 1

        if not dry_run:
            for dup in members:
                if dup.id == canonical_id:
                    continue
                if session.get(EntityRow, dup.id) is None:
                    continue
                # Same concept, different wording — keep canonical's entity_type.
                merge_entity(session, canonical_id, dup.id, update_types=False)

    if not dry_run:
        session.commit()

    log.info("activity_dedup_complete", clusters_found=len(merge_clusters),
             entities_merged=entities_merged)
    return {
        "clusters_found": len(merge_clusters),
        "entities_merged": entities_merged,
        "activities_before": len(activities),
    }
