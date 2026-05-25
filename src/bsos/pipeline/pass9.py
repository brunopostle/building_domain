"""Pass 9 — Force Extraction with direction validation.

For each active component, space, or system entity: extract design pressures into
ForceRow records. Each force name must contain a direction-consistent qualifier
word; failures are written to pending_force_refs. Unresolved entity names in
'affects' are written to pending_entity_refs rather than discarded.

Activities, materials, and ifc_class entities are skipped — design forces (in the
Alexander sense) act on physical elements, spaces, and systems, not on construction
activities, raw materials, or IFC schema classes.
"""
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import structlog
from sqlmodel import Session, select

from bsos.llm.protocol import LLMProvider
from bsos.pipeline._name_utils import normalized_forms
from bsos.persistence.models import (
    EntityAliasRow, EntityRow, ForceRow, PassProgressRow,
    PendingEntityRefRow, PendingForceRefRow,
)
from bsos.pipeline.schemas import ForceExtractionResponse

log = structlog.get_logger()

INCREASE_QUALIFIERS: frozenset[str] = frozenset({
    "increased", "improved", "enhanced", "maximised", "maximized",
    "greater", "higher", "more", "better", "stronger", "expanded",
    "adequate", "sufficient", "optimised", "optimized",
})

DECREASE_QUALIFIERS: frozenset[str] = frozenset({
    "reduced", "minimised", "minimized", "decreased", "limited",
    "lower", "less", "fewer", "smaller", "restricted", "constrained",
    "avoided", "prevented", "eliminated",
})

PROMPT_TEMPLATE = (
    "For the building entity '{name}' (type: {entity_type}), identify individual "
    "design pressures that act on decisions involving '{name}'.\n\n"
    "For each force:\n"
    "1. Name it as a directional pressure.\n"
    "   - For 'increase' direction, the name MUST contain one of: "
    "increased, improved, enhanced, maximised, maximized, greater, higher, more, "
    "better, stronger, expanded, adequate, sufficient, optimised, optimized.\n"
    "   - For 'decrease' direction, the name MUST contain one of: "
    "reduced, minimised, minimized, decreased, limited, lower, less, fewer, "
    "smaller, restricted, constrained, avoided, prevented, eliminated.\n"
    "2. Specify the direction: 'increase' or 'decrease'.\n"
    "3. List the building entities this force acts upon.\n"
    "4. Provide confidence (0-1), knowledge_origin "
    "(physical/engineering/architectural/cultural), and rationale."
)


def _validate_direction(name: str, direction: str) -> bool:
    """Return True if name contains a qualifier word consistent with direction."""
    name_lower = name.lower()
    qualifiers = INCREASE_QUALIFIERS if direction == "increase" else DECREASE_QUALIFIERS
    return any(q in name_lower for q in qualifiers)


def _build_name_lookup(engine) -> dict[str, str]:
    """Return lowercase_name → entity_id for all active entities and aliases.

    Also indexes plural/singular variants. Exact names always take priority.
    """
    lookup: dict[str, str] = {}
    with Session(engine) as s:
        for row in s.exec(select(EntityRow).where(EntityRow.status != "merged")).all():
            key = row.name.lower()
            lookup[key] = row.id
            for alt in normalized_forms(key):
                lookup.setdefault(alt, row.id)
        for alias_row in s.exec(select(EntityAliasRow)).all():
            entity = s.get(EntityRow, alias_row.entity_id)
            if entity and entity.status != "merged":
                key = alias_row.alias.lower()
                lookup[key] = entity.id
                for alt in normalized_forms(key):
                    lookup.setdefault(alt, entity.id)
    return lookup


def _process_entity(
    engine,
    entity_id: str,
    entity_name: str,
    entity_type: str,
    provider: LLMProvider,
    name_lookup: dict[str, str],
    run_id: str,
) -> tuple[int, int, int]:
    """Extract and write forces for one entity.

    Returns (forces_written, validation_failures, unresolved_refs).
    """
    with Session(engine) as session:
        progress = session.get(PassProgressRow, ("9", entity_id, provider.model_id))
        if progress and progress.status == "completed":
            log.debug("pass9_entity_skip_resume", entity=entity_name)
            return 0, 0, 0

        prompt = PROMPT_TEMPLATE.format(name=entity_name, entity_type=entity_type)
        try:
            response = provider.extract(
                prompt, ForceExtractionResponse, entity_name=entity_name
            )
        except Exception as exc:
            log.warning("pass9_extraction_failed", entity=entity_name, error=str(exc))
            return 0, 0, 0

        now = datetime.now(timezone.utc)
        forces_written = 0
        validation_failures = 0
        unresolved_refs = 0

        for extracted in response.forces:
            if not extracted.name or not extracted.name.strip():
                continue

            # Direction consistency validation
            if not _validate_direction(extracted.name, extracted.direction):
                log.warning(
                    "pass9_direction_validation_failure",
                    force_name=extracted.name,
                    direction=extracted.direction,
                    entity=entity_name,
                )
                session.add(PendingForceRefRow(
                    description=extracted.name,
                    failure_type="validation_failure",
                    created_at=now,
                ))
                validation_failures += 1
                continue

            # Resolve affects entity names
            resolved_ids: list[str] = []
            unresolved_names: list[str] = []
            for affect_name in extracted.affects:
                key = affect_name.lower().strip()
                if not key:
                    continue
                entity_id_found = name_lookup.get(key)
                if entity_id_found:
                    resolved_ids.append(entity_id_found)
                else:
                    unresolved_names.append(affect_name)

            force_id = str(uuid.uuid4())
            session.add(ForceRow(
                id=force_id,
                name=extracted.name.strip(),
                direction=extracted.direction,
                affects=json.dumps(resolved_ids),
                source_model=provider.model_id,
                source_prompt=prompt,
                created_at=now,
                extraction_run_id=run_id,
                confidence=round(extracted.confidence, 4),
                status="proposed",
                knowledge_origin=extracted.knowledge_origin,
                rationale=extracted.rationale or None,
            ))
            session.flush()

            for unresolved_name in unresolved_names:
                session.add(PendingEntityRefRow(
                    entity_name=unresolved_name,
                    source_force_id=force_id,
                    created_at=now,
                ))
                unresolved_refs += 1

            forces_written += 1

        existing_progress = session.get(PassProgressRow, ("9", entity_id, provider.model_id))
        if existing_progress:
            existing_progress.completed_at = now
            existing_progress.status = "completed"
        else:
            session.add(PassProgressRow(
                pass_number="9",
                entity_id=entity_id,
                model=provider.model_id,
                completed_at=now,
                status="completed",
            ))

        session.commit()
        log.info(
            "pass9_entity_done",
            entity=entity_name,
            forces_written=forces_written,
            validation_failures=validation_failures,
            unresolved_refs=unresolved_refs,
        )
        return forces_written, validation_failures, unresolved_refs


ENTITY_TYPES = frozenset({"component", "space", "system"})


def run_pass9(
    engine,
    provider: LLMProvider,
    run_id: str,
    dry_run: bool = False,
    max_workers: int = 4,
) -> dict:
    """Run Pass 9: force extraction for component, space, and system entities.

    Returns summary dict: {entities_processed, forces_written,
                           validation_failures, unresolved_refs}.
    """
    name_lookup = _build_name_lookup(engine)

    with Session(engine) as session:
        entities = session.exec(
            select(EntityRow).where(
                EntityRow.status != "merged",
                EntityRow.entity_type.in_(ENTITY_TYPES),  # type: ignore[attr-defined]
            )
        ).all()
        entity_tuples = [(e.id, e.name, e.entity_type) for e in entities]

    log.info("pass9_start", entity_count=len(entity_tuples))

    if dry_run:
        return {
            "entities_processed": len(entity_tuples),
            "forces_written": 0,
            "validation_failures": 0,
            "unresolved_refs": 0,
        }

    total_forces = 0
    total_failures = 0
    total_unresolved = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _process_entity,
                engine, eid, ename, etype, provider, name_lookup, run_id,
            ): ename
            for eid, ename, etype in entity_tuples
        }
        for future in as_completed(futures):
            entity_name = futures[future]
            try:
                written, failures, unresolved = future.result()
                total_forces += written
                total_failures += failures
                total_unresolved += unresolved
            except Exception as exc:
                log.error("pass9_entity_error", entity=entity_name, error=str(exc))

    log.info(
        "pass9_complete",
        entities_processed=len(entity_tuples),
        forces_written=total_forces,
        validation_failures=total_failures,
        unresolved_refs=total_unresolved,
    )
    return {
        "entities_processed": len(entity_tuples),
        "forces_written": total_forces,
        "validation_failures": total_failures,
        "unresolved_refs": total_unresolved,
    }
