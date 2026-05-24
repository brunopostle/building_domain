"""Pass 2 — Entity Deduplication via embedding-based clustering.

Computes all-mpnet-base-v2 embeddings for entity names, clusters with
AgglomerativeClustering (cosine, average linkage, distance_threshold=0.20),
then merges near-duplicate entities: elects canonical, inserts aliases,
re-points assertion FKs, marks duplicates status='merged'.
"""
import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import structlog
from sqlmodel import Session, select, func

from bsos.persistence.models import (
    AssertionRow, EmbeddingRow, EntityAliasRow, EntityRow, PassProgressRow,
)

log = structlog.get_logger()

EMBEDDING_MODEL = "all-mpnet-base-v2"
CLUSTER_DISTANCE_THRESHOLD = 0.04


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _load_or_compute_embeddings(
    session: Session,
    entities: list[EntityRow],
    embedding_model: str,
    embedder: Callable[[list[str]], np.ndarray],
) -> np.ndarray:
    vectors: dict[str, np.ndarray] = {}
    to_compute: list[EntityRow] = []

    for entity in entities:
        chash = _content_hash(entity.name)
        row = session.get(EmbeddingRow, ("entity", entity.id, embedding_model))
        if row and row.content_hash == chash:
            vectors[entity.id] = np.frombuffer(row.vector, dtype=np.float32).copy()
        else:
            to_compute.append(entity)

    if to_compute:
        names = [e.name for e in to_compute]
        computed = np.array(embedder(names), dtype=np.float32)
        for i, entity in enumerate(to_compute):
            chash = _content_hash(entity.name)
            vec = computed[i]
            existing = session.get(EmbeddingRow, ("entity", entity.id, embedding_model))
            if existing:
                existing.vector = vec.tobytes()
                existing.content_hash = chash
                existing.dim = int(vec.shape[0])
            else:
                session.add(EmbeddingRow(
                    item_type="entity",
                    item_id=entity.id,
                    model=embedding_model,
                    dim=int(vec.shape[0]),
                    content_hash=chash,
                    vector=vec.tobytes(),
                ))
            vectors[entity.id] = vec
        session.commit()

    return np.array([vectors[e.id] for e in entities], dtype=np.float32)


def _assertion_count(session: Session, entity_id: str) -> int:
    return session.exec(
        select(func.count(AssertionRow.id)).where(
            (AssertionRow.subject_id == entity_id) | (AssertionRow.object_id == entity_id)
        )
    ).one()


def _elect_canonical(session: Session, entity_ids: list[str]) -> str:
    counts = {eid: _assertion_count(session, eid) for eid in entity_ids}
    max_count = max(counts.values())
    candidates = [eid for eid in entity_ids if counts[eid] == max_count]

    if len(candidates) == 1:
        return candidates[0]

    rows = {
        e.id: e
        for e in session.exec(
            select(EntityRow).where(EntityRow.id.in_(candidates))
        ).all()
    }
    return min(candidates, key=lambda eid: rows[eid].created_at)


def _merge_cluster(session: Session, canonical_id: str, duplicate_ids: list[str]) -> None:
    for dup_id in duplicate_ids:
        dup = session.get(EntityRow, dup_id)
        if dup is None:
            continue

        session.add(EntityAliasRow(entity_id=canonical_id, alias=dup.name))

        for row in session.exec(
            select(AssertionRow).where(AssertionRow.subject_id == dup_id)
        ).all():
            row.subject_id = canonical_id

        for row in session.exec(
            select(AssertionRow).where(AssertionRow.object_id == dup_id)
        ).all():
            row.object_id = canonical_id

        dup.status = "merged"

    session.commit()


def _record_progress(session: Session, model: str) -> None:
    now = datetime.now(timezone.utc)
    existing = session.get(PassProgressRow, ("2", "__pass2__", model))
    if existing:
        existing.completed_at = now
        existing.status = "completed"
    else:
        session.add(PassProgressRow(
            pass_number="2",
            entity_id="__pass2__",
            model=model,
            completed_at=now,
            status="completed",
        ))
    session.commit()


def run_pass2(
    session: Session,
    run_id: str,
    embedding_model: str = EMBEDDING_MODEL,
    _embedder: Callable[[list[str]], np.ndarray] | None = None,
    dry_run: bool = False,
) -> dict:
    """Run Pass 2: compute embeddings, cluster, merge near-duplicate entities.

    Returns summary dict with keys: clusters_found, entities_merged.
    _embedder is a test seam; omit in production to use SentenceTransformer.
    """
    from sklearn.cluster import AgglomerativeClustering

    log.info("pass2_start", embedding_model=embedding_model)

    entities = session.exec(
        select(EntityRow).where(EntityRow.status != "merged")
    ).all()

    if len(entities) < 2:
        log.info("pass2_skip", reason="fewer than 2 active entities", count=len(entities))
        if not dry_run:
            _record_progress(session, embedding_model)
        return {"clusters_found": 0, "entities_merged": 0}

    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        model_obj = SentenceTransformer(embedding_model)
        embedder: Callable[[list[str]], np.ndarray] = lambda texts: model_obj.encode(
            texts, show_progress_bar=False
        )
    else:
        embedder = _embedder

    vectors = _load_or_compute_embeddings(session, entities, embedding_model, embedder)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=CLUSTER_DISTANCE_THRESHOLD,
    )
    labels = clustering.fit_predict(vectors)

    clusters: dict[int, list[EntityRow]] = defaultdict(list)
    for entity, label in zip(entities, labels):
        clusters[label].append(entity)

    merge_clusters = [members for members in clusters.values() if len(members) >= 2]
    log.info("pass2_clusters", total_clusters=len(clusters), merge_clusters=len(merge_clusters))

    entities_merged = 0
    for members in merge_clusters:
        ids = [m.id for m in members]
        canonical_id = _elect_canonical(session, ids)
        duplicate_ids = [eid for eid in ids if eid != canonical_id]
        canonical_name = next(m.name for m in members if m.id == canonical_id)
        dup_names = [m.name for m in members if m.id != canonical_id]
        log.info("pass2_merge", canonical=canonical_name, duplicates=dup_names)
        entities_merged += len(duplicate_ids)

        if not dry_run:
            _merge_cluster(session, canonical_id, duplicate_ids)

    if not dry_run:
        _record_progress(session, embedding_model)

    log.info("pass2_complete", clusters_found=len(merge_clusters), entities_merged=entities_merged)
    return {"clusters_found": len(merge_clusters), "entities_merged": entities_merged}
