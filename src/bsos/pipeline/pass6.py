"""Pass 6 — Constraint Extraction.

For each active entity: extract hard binary rules (must/must_not) into
ConstraintRow records. Only binary constraints where violation makes the design
physically invalid, unsafe, or inoperable are captured — not typical functional
dependencies or preferences.
"""
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import structlog
from sqlmodel import Session, select

from bsos.llm.protocol import LLMProvider
from bsos.persistence.models import ConstraintRow, EntityRow, PassProgressRow
from bsos.pipeline.schemas import ConstraintExtractionResponse

log = structlog.get_logger()

PROMPT_TEMPLATE = (
    "For the building entity '{name}' (type: {entity_type}), list only hard binary "
    "design constraints — rules where violation makes the design physically invalid, "
    "unsafe, or inoperable. Each constraint must be true or false with no middle ground.\n\n"
    "CONSTRAINT (binary, no valid exception):\n"
    "  'roof must have a drainage path'\n\n"
    "NOT A CONSTRAINT (dependency with form variation — use Assertion instead):\n"
    "  'roof requires structural support'\n\n"
    "For each constraint specify: the rule text, type ('must' or 'must_not'), "
    "any conditions that narrow its applicability, any genuine exceptions, "
    "confidence (0-1), knowledge_origin (physical/engineering/architectural/cultural), "
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
    """Extract and write constraints for one entity. Returns count written."""
    with Session(engine) as session:
        progress = session.get(PassProgressRow, ("6", entity_id, provider.model_id))
        if progress and progress.status == "completed":
            log.debug("pass6_entity_skip_resume", entity=entity_name)
            return 0

        prompt = PROMPT_TEMPLATE.format(name=entity_name, entity_type=entity_type)
        try:
            response = provider.extract(
                prompt, ConstraintExtractionResponse, entity_name=entity_name
            )
        except Exception as exc:
            log.warning("pass6_extraction_failed", entity=entity_name, error=str(exc))
            response = ConstraintExtractionResponse(constraints=[])

        now = datetime.now(timezone.utc)
        written = 0

        for extracted in response.constraints:
            if not extracted.rule or not extracted.rule.strip():
                continue

            session.add(ConstraintRow(
                id=str(uuid.uuid4()),
                subject_id=entity_id,
                rule=extracted.rule.strip(),
                constraint_type=extracted.constraint_type,
                conditions=json.dumps(extracted.conditions),
                exceptions=json.dumps(extracted.exceptions),
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

        existing_progress = session.get(PassProgressRow, ("6", entity_id, provider.model_id))
        if existing_progress:
            existing_progress.completed_at = now
            existing_progress.status = "completed"
        else:
            session.add(PassProgressRow(
                pass_number="6",
                entity_id=entity_id,
                model=provider.model_id,
                completed_at=now,
                status="completed",
            ))

        session.commit()
        log.info("pass6_entity_done", entity=entity_name, constraints_written=written)
        return written


def run_pass6(
    engine,
    provider: LLMProvider,
    run_id: str,
    dry_run: bool = False,
    max_workers: int = 4,
) -> dict:
    """Run Pass 6: constraint extraction for all active entities.

    Returns summary dict: {entities_processed, constraints_written}.
    """
    with Session(engine) as session:
        entities = session.exec(
            select(EntityRow).where(EntityRow.status != "merged")
        ).all()
        entity_tuples = [(e.id, e.name, e.entity_type) for e in entities]

    log.info("pass6_start", entity_count=len(entity_tuples))

    if dry_run:
        return {"entities_processed": len(entity_tuples), "constraints_written": 0}

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
                log.error("pass6_entity_error", entity=entity_name, error=str(exc))

    log.info("pass6_complete", entities_processed=len(entity_tuples), constraints_written=total_written)
    return {"entities_processed": len(entity_tuples), "constraints_written": total_written}
