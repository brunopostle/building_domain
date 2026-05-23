"""Pass 10c — Abstraction synthesis.

Algorithm per Section 8.4:
  1. Group accepted/proposed assertions by subject_id.
  2. For each group with ≥ min_cluster_size assertions: embed texts, cluster.
  3. For each qualifying cluster (≥ min_cluster_size):
     a. LLM-A synthesizes one statement capturing the cluster.
     b. LLM-B (adversarial) validates it does not introduce new information.
        If LLM-B unavailable or same model as LLM-A: skip validation, accept.
        If LLM-B says "yes" (introduces new info): discard this cluster's node.
     c. Create AbstractionNode(status='proposed') with child_ids = cluster assertion UUIDs.
  4. Stop if the 200-node proposed queue cap is reached.
  5. Graph-layer aggregate edges are deferred to the graph-construction pass.

Completion is tracked in pass_progress with key '10c'.
"""
import json
import uuid
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import structlog
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from bsos.llm.protocol import LLMProvider
from bsos.models.abstraction import AbstractionNode
from bsos.persistence.database import create_views
from bsos.persistence.models import AssertionRow, EntityRow, PassProgressRow
from bsos.persistence.repos.abstraction import AbstractionNodeRepository

log = structlog.get_logger()

EMBEDDING_MODEL = "all-mpnet-base-v2"
MIN_CLUSTER_SIZE = 3
CLUSTER_DISTANCE_THRESHOLD = 0.25  # cosine distance; equiv. to cos_sim ≥ 0.75
ABSTRACTION_QUEUE_CAP = 200


# ---------------------------------------------------------------------------
# LLM schemas
# ---------------------------------------------------------------------------

class _SynthesisResponse(BaseModel):
    statement: str = Field(description="Single statement capturing the cluster without new information")
    abstraction_rationale: str = Field(description="Why this statement faithfully summarises the cluster")
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assertion_text(row: AssertionRow) -> str:
    """Build a short semantic text string for embedding."""
    parts = [row.predicate]
    conds = json.loads(row.conditions or "[]")
    if conds:
        parts.append("; ".join(conds))
    if row.rationale:
        parts.append(row.rationale)
    return " | ".join(parts)


def _load_entity_names(session: Session) -> dict[str, str]:
    return {r.id: r.name for r in session.exec(select(EntityRow)).all()}


def _load_grouped_assertions(session: Session) -> dict[str, list[AssertionRow]]:
    """Return subject_id → list of accepted/proposed AssertionRows."""
    rows = session.exec(
        select(AssertionRow).where(AssertionRow.status.in_(["accepted", "proposed"]))
    ).all()
    groups: dict[str, list[AssertionRow]] = {}
    for r in rows:
        groups.setdefault(r.subject_id, []).append(r)
    return groups


def _cosine_distance_matrix(vecs: np.ndarray) -> np.ndarray:
    """Return (N×N) pairwise cosine distance matrix."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normed = np.where(norms > 0, vecs / norms, 0.0)
    sims = normed @ normed.T
    sims = np.clip(sims, -1.0, 1.0)
    return 1.0 - sims


def _cluster(vecs: np.ndarray, threshold: float) -> list[int]:
    """Hierarchical clustering; returns cluster label per assertion (1-indexed)."""
    from scipy.cluster.hierarchy import fclusterdata
    if len(vecs) < 2:
        return [1] * len(vecs)
    return fclusterdata(vecs, t=threshold, criterion="distance", metric="cosine", method="average").tolist()


def _group_by_label(labels: list[int]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        groups.setdefault(label, []).append(i)
    return groups


def _build_synthesis_prompt(rows: list[AssertionRow], entity_names: dict[str, str]) -> str:
    lines = [
        "You are analyzing a cluster of related building-domain assertions that share the same subject entity.",
        "",
        "Source assertions:",
    ]
    for i, row in enumerate(rows, 1):
        subj = entity_names.get(row.subject_id, row.subject_id)
        obj = entity_names.get(row.object_id, row.object_id)
        line = f"  {i}. {subj} {row.predicate} {obj}"
        if row.rationale:
            line += f" — {row.rationale}"
        conds = json.loads(row.conditions or "[]")
        if conds:
            line += f" [when: {'; '.join(conds)}]"
        lines.append(line)
    lines += [
        "",
        "Write a single statement that captures the essential meaning of all these assertions "
        "without introducing any new information not already present in the sources.",
    ]
    return "\n".join(lines)


def _build_validation_prompt(cluster_text: str, statement: str) -> str:
    return (
        f"Source assertions:\n{cluster_text}\n\n"
        f"Proposed abstraction:\n  {statement}\n\n"
        "Does this abstraction assert or imply anything NOT already explicitly present "
        "in the source assertions?\n"
        "Answer 'yes' if it introduces new claims. Answer 'no' if it only captures what is stated."
    )


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def _process_cluster(
    engine,
    cluster_rows: list[AssertionRow],
    entity_names: dict[str, str],
    provider_a: LLMProvider,
    provider_b: LLMProvider | None,
    run_id: str,
    now: datetime,
) -> bool:
    """Synthesize one AbstractionNode for a cluster. Returns True if node was created."""
    synthesis_prompt = _build_synthesis_prompt(cluster_rows, entity_names)

    try:
        response = provider_a.extract(synthesis_prompt, _SynthesisResponse)
    except Exception as exc:
        log.warning("pass10c_synthesis_failed", error=str(exc), cluster_size=len(cluster_rows))
        return False

    # Adversarial validation
    use_validation = (
        provider_b is not None
        and provider_b.model_id != provider_a.model_id
    )
    if use_validation:
        cluster_lines = "\n".join(
            f"  {i+1}. {r.predicate}: {r.rationale or ''}" for i, r in enumerate(cluster_rows)
        )
        val_prompt = _build_validation_prompt(cluster_lines, response.statement)
        try:
            verdict = provider_b.classify(val_prompt, ["yes", "no"])
            if verdict.strip().lower() == "yes":
                log.info(
                    "pass10c_abstraction_rejected",
                    statement=response.statement,
                    reason="adversarial_validation",
                )
                return False
        except Exception as exc:
            log.warning("pass10c_validation_failed", error=str(exc))
            # Fall through — create node without validation

    child_ids = [r.id for r in cluster_rows]
    source_model = provider_a.model_id
    node = AbstractionNode(
        id=str(uuid.uuid4()),
        statement=response.statement,
        child_ids=child_ids,
        abstraction_rationale=response.abstraction_rationale,
        source_model=source_model,
        source_prompt=synthesis_prompt,
        created_at=now,
        extraction_run_id=run_id,
        confidence=response.confidence,
        status="proposed",
        rationale=None,
        conflict_evaluated_at=None,
    )

    with Session(engine) as session:
        repo = AbstractionNodeRepository(session)
        repo.add(node)
        session.commit()

    log.info(
        "pass10c_node_created",
        node_id=node.id,
        cluster_size=len(cluster_rows),
        statement=response.statement[:80],
    )
    # Graph-layer aggregate edges deferred to the graph-construction pass.
    return True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pass10c(
    engine,
    provider_a: LLMProvider,
    provider_b: LLMProvider | None = None,
    embedding_model: str = EMBEDDING_MODEL,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    cluster_distance_threshold: float = CLUSTER_DISTANCE_THRESHOLD,
    _embedder: Callable[[list[str]], np.ndarray] | None = None,
    dry_run: bool = False,
) -> dict:
    """Run Pass 10c: abstraction synthesis.

    provider_a is used for cluster synthesis. provider_b (if distinct from
    provider_a) performs adversarial validation; omit to skip validation.
    _embedder is a test seam; omit in production to use SentenceTransformer.
    """
    with Session(engine) as session:
        progress = session.get(PassProgressRow, ("10c", "__global__", embedding_model))
        if progress and progress.status == "completed":
            log.info("pass10c_skip", reason="already completed")
            return {"status": "already_completed"}

    log.info("pass10c_start", embedding_model=embedding_model, min_cluster_size=min_cluster_size)

    with Session(engine) as session:
        subject_groups = _load_grouped_assertions(session)
        entity_names = _load_entity_names(session)

    # Prune groups that cannot produce any qualifying cluster.
    eligible_groups = {
        sid: rows for sid, rows in subject_groups.items()
        if len(rows) >= min_cluster_size
    }

    if dry_run:
        return {
            "dry_run": True,
            "eligible_subject_groups": len(eligible_groups),
            "total_assertions_in_scope": sum(len(v) for v in eligible_groups.values()),
        }

    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _st = SentenceTransformer(embedding_model)
        embedder: Callable[[list[str]], np.ndarray] = lambda texts: _st.encode(
            texts, show_progress_bar=False
        )
    else:
        embedder = _embedder

    run_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    nodes_created = 0
    clusters_processed = 0
    cap_reached = False

    for subject_id, rows in eligible_groups.items():
        # Check cap before processing each subject group.
        with Session(engine) as session:
            repo = AbstractionNodeRepository(session)
            proposed_count = repo.count_proposed()
        if proposed_count >= ABSTRACTION_QUEUE_CAP:
            log.warning(
                "pass10c_cap_reached",
                cap=ABSTRACTION_QUEUE_CAP,
                proposed_count=proposed_count,
            )
            cap_reached = True
            break

        texts = [_assertion_text(r) for r in rows]
        vecs = np.array(embedder(texts), dtype=np.float32)

        labels = _cluster(vecs, cluster_distance_threshold)
        cluster_groups = _group_by_label(labels)

        for label, indices in cluster_groups.items():
            if len(indices) < min_cluster_size:
                continue

            # Re-check cap before each node.
            with Session(engine) as session:
                repo = AbstractionNodeRepository(session)
                proposed_count = repo.count_proposed()
            if proposed_count >= ABSTRACTION_QUEUE_CAP:
                log.warning(
                    "pass10c_cap_reached",
                    cap=ABSTRACTION_QUEUE_CAP,
                    proposed_count=proposed_count,
                )
                cap_reached = True
                break

            cluster_rows = [rows[i] for i in indices]
            clusters_processed += 1
            created = _process_cluster(
                engine, cluster_rows, entity_names,
                provider_a, provider_b, run_id, now,
            )
            if created:
                nodes_created += 1

        if cap_reached:
            break

    with Session(engine) as session:
        now_complete = datetime.now(timezone.utc)
        existing = session.get(PassProgressRow, ("10c", "__global__", embedding_model))
        if existing:
            existing.completed_at = now_complete
            existing.status = "completed"
        else:
            session.add(PassProgressRow(
                pass_number="10c",
                entity_id="__global__",
                model=embedding_model,
                completed_at=now_complete,
                status="completed",
            ))
        session.commit()

    result = {
        "eligible_subject_groups": len(eligible_groups),
        "clusters_processed": clusters_processed,
        "nodes_created": nodes_created,
        "cap_reached": cap_reached,
    }
    log.info("pass10c_complete", **result)
    return result
