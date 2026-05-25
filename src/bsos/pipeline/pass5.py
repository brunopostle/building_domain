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
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import structlog
from sqlmodel import Session, select

from bsos.llm.protocol import LLMProvider
from bsos.persistence.models import EntityAliasRow, EntityRow, PassProgressRow, ProcessRelationRow
from bsos.pipeline.schemas import ProcessRelationExtractionResponse

log = structlog.get_logger()

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
