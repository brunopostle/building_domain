"""Pass 1 — Concept Discovery.

Accepts a seed (user list, free-text, or default bootstrap), calls the LLM to
enumerate building concepts including construction activities, expands each
top-level concept for sub-concepts, then writes Entity rows.
"""
import uuid
from datetime import datetime, timezone
from typing import Literal

import structlog
from sqlmodel import Session, select

from bsos.llm.protocol import LLMProvider
from bsos.persistence.models import EntityRow, PassProgressRow
from bsos.pipeline.schemas import (
    ConceptDiscoveryResponse, ConceptExpansionResponse, DiscoveredConcept,
)

log = structlog.get_logger()

BOOTSTRAP_PROMPT = (
    "List all major building concepts found in a typical building, covering:\n"
    "- structural and envelope components (roof, wall, foundation, beam, column, slab, window, door)\n"
    "- building systems (HVAC, plumbing, electrical, fire suppression, lifts)\n"
    "- spaces (corridor, stairwell, plant room, car park, office, toilet, lobby, atrium)\n"
    "- materials (concrete, timber, steel, glass, insulation, waterproof membrane, brick)\n"
    "- construction activities (waterproofing, formwork installation, formwork removal, "
    "concrete pouring, curing, finishing, glazing, painting, testing and commissioning)\n\n"
    "For each concept provide its name, type (component/system/space/material/activity), "
    "and a one-sentence description. Be comprehensive — include sub-systems and activities "
    "that are prerequisites for later construction stages."
)

DOMAIN_PROMPT_TEMPLATE = (
    "List all building concepts relevant to the following domain:\n\n"
    "{domain}\n\n"
    "Cover components, systems, spaces, materials, and construction activities. "
    "For each concept provide its name, type (component/system/space/material/activity), "
    "and a one-sentence description."
)

EXPANSION_PROMPT_TEMPLATE = (
    "For the building concept '{name}' (type: {entity_type}), list all sub-components, "
    "variants, related construction activities, and closely related concepts that should be "
    "tracked separately in a building knowledge system. "
    "For each sub-concept provide its name, type (component/system/space/material/activity), "
    "and a one-sentence description. Omit concepts already covered by the parent name."
)

APL_SEED_PROMPT = "apl_pattern_seed"


def run_pass1(
    session: Session,
    provider: LLMProvider,
    run_id: str,
    seed: str | None = None,
    seed_is_file_contents: bool = False,
    apl_patterns: list[str] | None = None,
    dry_run: bool = False,
) -> list[DiscoveredConcept]:
    """Run Pass 1. Returns the list of discovered concepts (for dry-run reporting).

    seed: free-text domain description, or None for default bootstrap.
    seed_is_file_contents: if True, seed is a newline-separated concept list;
                           skip discovery and go straight to expansion.
    apl_patterns: optional list of title-cased Alexander pattern names to merge as seeds.
    """
    log.info("pass1_start", model=provider.model_id, seed_provided=seed is not None)

    top_level: list[DiscoveredConcept] = []

    if seed_is_file_contents and seed:
        # User-supplied concept list — infer types via a single LLM call then expand
        names = [line.strip() for line in seed.splitlines() if line.strip()]
        top_level = [
            DiscoveredConcept(name=name, entity_type="component") for name in names
        ]
        log.info("pass1_seed_list", concept_count=len(top_level))
    else:
        # LLM discovery
        prompt = DOMAIN_PROMPT_TEMPLATE.format(domain=seed) if seed else BOOTSTRAP_PROMPT
        entity_name = seed[:40].strip() if seed else "__bootstrap__"
        response = provider.extract(prompt, ConceptDiscoveryResponse, entity_name=entity_name)
        top_level = response.concepts  # type: ignore[attr-defined]
        log.info("pass1_discovery", concept_count=len(top_level))

    if apl_patterns:
        existing_lower = {c.name.lower() for c in top_level}
        added = 0
        for name in apl_patterns:
            if name.lower() not in existing_lower:
                top_level.append(DiscoveredConcept(name=name, entity_type="space"))
                existing_lower.add(name.lower())
                added += 1
        log.info("pass1_apl_merged", apl_requested=len(apl_patterns), apl_added=added)

    # Expand each top-level concept for sub-concepts
    all_concepts: list[DiscoveredConcept] = list(top_level)
    for concept in top_level:
        expansion_prompt = EXPANSION_PROMPT_TEMPLATE.format(
            name=concept.name, entity_type=concept.entity_type
        )
        try:
            expansion = provider.extract(
                expansion_prompt, ConceptExpansionResponse, entity_name=concept.name
            )
            all_concepts.extend(expansion.sub_concepts)  # type: ignore[attr-defined]
        except Exception as exc:
            log.warning("pass1_expansion_failed", concept=concept.name, error=str(exc))

    # Deduplicate by name (case-insensitive)
    seen: dict[str, DiscoveredConcept] = {}
    for c in all_concepts:
        key = c.name.lower().strip()
        if key not in seen:
            seen[key] = c
    unique_concepts = list(seen.values())
    log.info("pass1_unique_concepts", count=len(unique_concepts))

    if dry_run:
        return unique_concepts

    # Write to DB — skip names already present (idempotent)
    existing_names = {
        row.name.lower()
        for row in session.exec(select(EntityRow)).all()
    }

    apl_names_lower = {n.lower() for n in (apl_patterns or [])}
    default_prompt = BOOTSTRAP_PROMPT if not seed else DOMAIN_PROMPT_TEMPLATE.format(domain=seed)

    now = datetime.now(timezone.utc)
    new_count = 0
    for concept in unique_concepts:
        if concept.name.lower() in existing_names:
            continue
        row = EntityRow(
            id=str(uuid.uuid4()),
            name=concept.name,
            entity_type=concept.entity_type,
            description=concept.description,
            status="proposed",
            source_model=provider.model_id,
            source_prompt=APL_SEED_PROMPT if concept.name.lower() in apl_names_lower else default_prompt,
            created_at=now,
            extraction_run_id=run_id,
        )
        session.add(row)
        existing_names.add(concept.name.lower())
        new_count += 1

    # Record pass completion (upsert — idempotent on re-run)
    existing_progress = session.get(PassProgressRow, ("1", "__pass1__", provider.model_id))
    if existing_progress:
        existing_progress.completed_at = now
        existing_progress.status = "completed"
    else:
        session.add(PassProgressRow(
            pass_number="1",
            entity_id="__pass1__",
            model=provider.model_id,
            completed_at=now,
            status="completed",
        ))
    session.commit()

    log.info("pass1_complete", new_entities=new_count, model=provider.model_id)
    return unique_concepts
