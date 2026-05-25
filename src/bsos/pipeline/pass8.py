"""Pass 8 — Pattern Extraction.

For each active component, space, or system entity: extract Alexander-style
architectural patterns into PatternRow records. force_ids and related_pattern_ids
are written as empty lists at extraction time and resolved in a later pass. Each
record is linked to the entity it was extracted for via subject_id.

Activities, materials, and ifc_class entities are skipped — Alexander patterns
describe recurring solutions to spatial/physical design problems, not construction
activities, raw materials, or IFC schema classes.
"""
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import structlog
from sqlmodel import Session, select

from bsos.llm.protocol import LLMProvider
from bsos.persistence.models import EntityRow, PassProgressRow, PatternRow
from bsos.pipeline.schemas import PatternExtractionResponse

log = structlog.get_logger()

PROMPT_TEMPLATE = (
    "For the building entity '{name}' (type: {entity_type}), describe architectural "
    "or spatial patterns that involve '{name}' and improve building quality.\n\n"
    "For each pattern provide:\n"
    "- name: a short descriptive name\n"
    "- context: list of situations where this pattern applies\n"
    "- problem: the recurring problem the pattern addresses\n"
    "- force_descriptions: the competing forces or tensions at play (free-text)\n"
    "- solution: the design solution the pattern proposes\n"
    "- consequences: what follows from applying this pattern\n"
    "- emergent_properties: qualities that emerge from using this pattern\n"
    "- related_pattern_names: names of other patterns that complement or conflict\n"
    "- confidence (0-1), knowledge_origin (physical/engineering/architectural/cultural), "
    "and rationale."
)


def _process_entity(
    engine,
    entity_id: str,
    entity_name: str,
    entity_type: str,
    provider: LLMProvider,
    run_id: str,
) -> int:
    """Extract and write patterns for one entity. Returns count written."""
    with Session(engine) as session:
        progress = session.get(PassProgressRow, ("8", entity_id, provider.model_id))
        if progress and progress.status == "completed":
            log.debug("pass8_entity_skip_resume", entity=entity_name)
            return 0

        prompt = PROMPT_TEMPLATE.format(name=entity_name, entity_type=entity_type)
        try:
            response = provider.extract(
                prompt, PatternExtractionResponse, entity_name=entity_name
            )
        except Exception as exc:
            log.warning("pass8_extraction_failed", entity=entity_name, error=str(exc))
            return 0

        now = datetime.now(timezone.utc)
        written = 0

        for extracted in response.patterns:
            if not extracted.name or not extracted.name.strip():
                continue
            if not extracted.problem or not extracted.problem.strip():
                continue
            if not extracted.solution or not extracted.solution.strip():
                continue

            session.add(PatternRow(
                id=str(uuid.uuid4()),
                name=extracted.name.strip(),
                subject_id=entity_id,
                context=json.dumps(extracted.context),
                problem=extracted.problem.strip(),
                force_descriptions=json.dumps(extracted.force_descriptions),
                force_ids=json.dumps([]),
                solution=extracted.solution.strip(),
                consequences=json.dumps(extracted.consequences),
                related_pattern_names=json.dumps(extracted.related_pattern_names),
                related_pattern_ids=json.dumps([]),
                emergent_properties=json.dumps(extracted.emergent_properties),
                source_model=provider.model_id,
                source_prompt=prompt,
                created_at=now,
                extraction_run_id=run_id,
                confidence=round(extracted.confidence, 4),
                status="proposed",
                knowledge_origin=extracted.knowledge_origin,
                rationale=extracted.rationale or None,
            ))
            written += 1

        existing_progress = session.get(PassProgressRow, ("8", entity_id, provider.model_id))
        if existing_progress:
            existing_progress.completed_at = now
            existing_progress.status = "completed"
        else:
            session.add(PassProgressRow(
                pass_number="8",
                entity_id=entity_id,
                model=provider.model_id,
                completed_at=now,
                status="completed",
            ))

        session.commit()
        log.info("pass8_entity_done", entity=entity_name, patterns_written=written)
        return written


ENTITY_TYPES = frozenset({"component", "space", "system"})


def run_pass8(
    engine,
    provider: LLMProvider,
    run_id: str,
    dry_run: bool = False,
    max_workers: int = 4,
) -> dict:
    """Run Pass 8: pattern extraction for component, space, and system entities.

    Returns summary dict: {entities_processed, patterns_written}.
    """
    with Session(engine) as session:
        entities = session.exec(
            select(EntityRow).where(
                EntityRow.status != "merged",
                EntityRow.entity_type.in_(ENTITY_TYPES),  # type: ignore[attr-defined]
            )
        ).all()
        entity_tuples = [(e.id, e.name, e.entity_type) for e in entities]

    log.info("pass8_start", entity_count=len(entity_tuples))

    if dry_run:
        return {"entities_processed": len(entity_tuples), "patterns_written": 0}

    total_written = 0

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
                total_written += future.result()
            except Exception as exc:
                log.error("pass8_entity_error", entity=entity_name, error=str(exc))

    log.info("pass8_complete", entities_processed=len(entity_tuples), patterns_written=total_written)
    return {"entities_processed": len(entity_tuples), "patterns_written": total_written}
