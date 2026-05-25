"""Pass 7 — Anti-Pattern Extraction.

For each active entity: extract known failure conditions, pathological
configurations, and design mistakes into AntiPatternRow records. Each record
is linked to the entity it was extracted for via subject_id.
"""
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import structlog
from sqlmodel import Session, select

from bsos.llm.protocol import LLMProvider
from bsos.persistence.models import AntiPatternRow, EntityRow, PassProgressRow
from bsos.pipeline.schemas import AntiPatternExtractionResponse

log = structlog.get_logger()

PROMPT_TEMPLATE = (
    "For the building entity '{name}' (type: {entity_type}), describe known failure "
    "conditions, pathological configurations, and design mistakes that consistently "
    "lead to physical, functional, or performance failures.\n\n"
    "Focus on recurring patterns seen in practice — not theoretical edge cases. "
    "For each anti-pattern provide: a short descriptive name, the conditions that "
    "cause it, the consequences (what fails and how), mitigations (how to avoid or "
    "recover), confidence (0-1), knowledge_origin "
    "(physical/engineering/architectural/cultural), and rationale."
)


def _process_entity(
    engine,
    entity_id: str,
    entity_name: str,
    entity_type: str,
    provider: LLMProvider,
    run_id: str,
) -> int:
    """Extract and write anti-patterns for one entity. Returns count written."""
    with Session(engine) as session:
        progress = session.get(PassProgressRow, ("7", entity_id, provider.model_id))
        if progress and progress.status == "completed":
            log.debug("pass7_entity_skip_resume", entity=entity_name)
            return 0

        prompt = PROMPT_TEMPLATE.format(name=entity_name, entity_type=entity_type)
        try:
            response = provider.extract(
                prompt, AntiPatternExtractionResponse, entity_name=entity_name
            )
        except Exception as exc:
            log.warning("pass7_extraction_failed", entity=entity_name, error=str(exc))
            return 0

        now = datetime.now(timezone.utc)
        written = 0

        for extracted in response.anti_patterns:
            if not extracted.name or not extracted.name.strip():
                continue

            session.add(AntiPatternRow(
                id=str(uuid.uuid4()),
                name=extracted.name.strip(),
                subject_id=entity_id,
                conditions=json.dumps(extracted.conditions),
                consequences=json.dumps(extracted.consequences),
                mitigations=json.dumps(extracted.mitigations),
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

        existing_progress = session.get(PassProgressRow, ("7", entity_id, provider.model_id))
        if existing_progress:
            existing_progress.completed_at = now
            existing_progress.status = "completed"
        else:
            session.add(PassProgressRow(
                pass_number="7",
                entity_id=entity_id,
                model=provider.model_id,
                completed_at=now,
                status="completed",
            ))

        session.commit()
        log.info("pass7_entity_done", entity=entity_name, anti_patterns_written=written)
        return written


def run_pass7(
    engine,
    provider: LLMProvider,
    run_id: str,
    dry_run: bool = False,
    max_workers: int = 4,
) -> dict:
    """Run Pass 7: anti-pattern extraction for all active entities.

    Returns summary dict: {entities_processed, anti_patterns_written}.
    """
    with Session(engine) as session:
        entities = session.exec(
            select(EntityRow).where(EntityRow.status != "merged")
        ).all()
        entity_tuples = [(e.id, e.name, e.entity_type) for e in entities]

    log.info("pass7_start", entity_count=len(entity_tuples))

    if dry_run:
        return {"entities_processed": len(entity_tuples), "anti_patterns_written": 0}

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
                log.error("pass7_entity_error", entity=entity_name, error=str(exc))

    log.info("pass7_complete", entities_processed=len(entity_tuples), anti_patterns_written=total_written)
    return {"entities_processed": len(entity_tuples), "anti_patterns_written": total_written}
