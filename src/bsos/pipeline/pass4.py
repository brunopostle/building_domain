"""Pass 4 — Spatial Relation Extraction.

For each active (non-merged) entity: ask the LLM what spatial and topological
relationships it has with other building entities. Relations in SPATIAL_RELATION_TYPES
are written directly; unknown types are also written to pending_spatial_relation_types
for curation. Unresolved object names are logged and skipped.
"""
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import structlog
from sqlmodel import Session, select

from bsos.llm.protocol import LLMProvider
from bsos.persistence.models import (
    EntityAliasRow, EntityRow, PassProgressRow,
    SpatialRelationRow,
)
from bsos.persistence.repos.pending import upsert_pending_spatial_relation_type
from bsos.pipeline.schemas import SpatialRelationExtractionResponse
from bsos.vocab import SPATIAL_RELATION_TYPES

log = structlog.get_logger()

_SPATIAL_TYPES_STR = ", ".join(SPATIAL_RELATION_TYPES)

PROMPT_TEMPLATE = (
    "For the building entity '{name}' (type: {entity_type}), list its spatial and topological "
    "relationships with other building entities. "
    "Preferred relation types: {spatial_types}. "
    "You may use other relation types if none of the above fit — they will be reviewed for vocabulary expansion. "
    "For each relationship provide: relation type, the other entity name, confidence (0-1), "
    "knowledge_origin (physical/engineering/architectural/cultural), and a brief rationale."
)


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
    run_id: str,
) -> int:
    """Extract and write spatial relations for one entity. Returns count written."""
    with Session(engine) as session:
        progress = session.get(PassProgressRow, ("4", entity_id, provider.model_id))
        if progress and progress.status == "completed":
            log.debug("pass4_entity_skip_resume", entity=entity_name)
            return 0

        prompt = PROMPT_TEMPLATE.format(
            name=entity_name,
            entity_type=entity_type,
            spatial_types=_SPATIAL_TYPES_STR,
        )
        try:
            response = provider.extract(
                prompt, SpatialRelationExtractionResponse, entity_name=entity_name
            )
        except Exception as exc:
            log.warning("pass4_extraction_failed", entity=entity_name, error=str(exc))
            response = SpatialRelationExtractionResponse(spatial_relations=[])

        now = datetime.now(timezone.utc)
        written = 0

        for extracted in response.spatial_relations:
            if not extracted.object_name.strip():
                continue

            obj_key = extracted.object_name.lower().strip()
            obj_info = name_lookup.get(obj_key)
            if obj_info is None:
                log.debug("pass4_unresolved_object", entity=entity_name, object=extracted.object_name)
                continue

            object_id, _ = obj_info
            if object_id == entity_id:
                continue

            if extracted.relation not in SPATIAL_RELATION_TYPES:
                upsert_pending_spatial_relation_type(session, extracted.relation)

            row = SpatialRelationRow(
                id=str(uuid.uuid4()),
                subject_id=entity_id,
                relation=extracted.relation,
                object_id=object_id,
                source_model=provider.model_id,
                source_prompt=prompt,
                created_at=now,
                extraction_run_id=run_id,
                confidence=round(extracted.confidence, 4),
                status="proposed",
                knowledge_origin=extracted.knowledge_origin,
                rationale=extracted.rationale or None,
            )
            session.add(row)
            written += 1

        existing_progress = session.get(PassProgressRow, ("4", entity_id, provider.model_id))
        if existing_progress:
            existing_progress.completed_at = now
            existing_progress.status = "completed"
        else:
            session.add(PassProgressRow(
                pass_number="4",
                entity_id=entity_id,
                model=provider.model_id,
                completed_at=now,
                status="completed",
            ))

        session.commit()
        log.info("pass4_entity_done", entity=entity_name, relations_written=written)
        return written


def run_pass4(
    engine,
    provider: LLMProvider,
    run_id: str,
    dry_run: bool = False,
    max_workers: int = 4,
) -> dict:
    """Run Pass 4: spatial relation extraction for all active entities.

    engine: SQLAlchemy Engine (not session) — workers create their own sessions.
    Returns summary dict: {entities_processed, relations_written}.
    """
    name_lookup = _build_name_lookup(engine)

    with Session(engine) as session:
        entities = session.exec(
            select(EntityRow).where(EntityRow.status != "merged")
        ).all()
        entity_tuples = [(e.id, e.name, e.entity_type) for e in entities]

    log.info("pass4_start", entity_count=len(entity_tuples))

    if dry_run:
        log.info("pass4_dry_run", entities=len(entity_tuples))
        return {"entities_processed": len(entity_tuples), "relations_written": 0}

    total_written = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _process_entity,
                engine, eid, ename, etype,
                provider, name_lookup, run_id,
            ): ename
            for eid, ename, etype in entity_tuples
        }
        for future in as_completed(futures):
            entity_name = futures[future]
            try:
                total_written += future.result()
            except Exception as exc:
                log.error("pass4_entity_error", entity=entity_name, error=str(exc))

    log.info("pass4_complete", entities_processed=len(entity_tuples), relations_written=total_written)
    return {"entities_processed": len(entity_tuples), "relations_written": total_written}
