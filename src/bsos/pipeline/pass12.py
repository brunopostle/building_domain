"""Pass 12 — IFC Schema Extraction.

Reads IFC source documents (PDF) from data/ and extracts:
  1. IFC class entities        → EntityRow(entity_type='ifc_class')
  2. Schema relationships      → AssertionRow(knowledge_origin='schema', confidence=1.0)
  3. Operational constraints   → ConstraintRow(knowledge_origin='schema', confidence=1.0)

Documents are processed in page chunks. Progress is tracked per chunk so the
pass is safely resumable.
"""
import json
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

import structlog
from sqlmodel import Session, select

from bsos.llm.protocol import LLMProvider
from bsos.persistence.models import (
    AssertionRow, ConstraintRow, EntityRow, PassProgressRow,
)
from bsos.pipeline.schemas import IFCChunkExtractionResponse

log = structlog.get_logger()

PAGES_PER_CHUNK = 5

IFC_SOURCES = [
    "ifc_implementation_guide_2x3_v2.pdf",
    "ifc_concepts_buildwise_2025.pdf",
]

CHUNK_PROMPT_TEMPLATE = (
    "You are reading an excerpt from an IFC (Industry Foundation Classes) specification "
    "document. Extract structured information from the following text.\n\n"
    "Extract:\n"
    "1. IFC class names mentioned (e.g. IfcWindow, IfcWall, IfcOpeningElement). "
    "Only include formally-named IFC schema classes (names starting with 'Ifc').\n"
    "2. Schema relationships between IFC classes — formal containment, "
    "connectivity, or dependency relationships defined by the spec "
    "(e.g. IfcProject contains IfcSite, IfcWindow requires IfcOpeningElement).\n"
    "3. Operational constraints — must/must_not rules for how IFC classes must be "
    "created, connected, or structured "
    "(e.g. IfcWall must have at least one IfcMaterialLayerSetUsage).\n\n"
    "Only extract information that is explicitly stated in the text below. "
    "Do not infer beyond what is written.\n\n"
    "--- DOCUMENT EXCERPT ---\n"
    "{text}\n"
    "--- END EXCERPT ---"
)


def _read_pdf_chunks(pdf_path: Path, pages_per_chunk: int) -> list[tuple[str, str]]:
    """Return list of (chunk_id, text) for every page group in the PDF."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pypdf  # noqa: PLC0415
        reader = pypdf.PdfReader(str(pdf_path))

    chunks: list[tuple[str, str]] = []
    pages = reader.pages
    for start in range(0, len(pages), pages_per_chunk):
        end = min(start + pages_per_chunk, len(pages))
        text_parts: list[str] = []
        for p in pages[start:end]:
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            if t.strip():
                text_parts.append(t)
        text = "\n\n".join(text_parts).strip()
        if text:
            chunk_id = f"{pdf_path.stem}_p{start}-{end - 1}"
            chunks.append((chunk_id, text))
    return chunks


def _get_or_create_entity(
    session: Session,
    name: str,
    description: str,
    provider: LLMProvider,
    run_id: str,
    now: datetime,
    name_cache: dict[str, str],
) -> str:
    """Return entity_id for an IFC class, creating it if absent."""
    key = name.lower().strip()
    if key in name_cache:
        return name_cache[key]

    existing = session.exec(
        select(EntityRow).where(EntityRow.name == name)
    ).first()
    if existing:
        name_cache[key] = existing.id
        return existing.id

    entity_id = str(uuid.uuid4())
    session.add(EntityRow(
        id=entity_id,
        name=name,
        entity_type="ifc_class",
        description=description,
        status="proposed",
        source_model=provider.model_id,
        source_prompt="ifc_schema_extraction",
        created_at=now,
        extraction_run_id=run_id,
    ))
    name_cache[key] = entity_id
    log.debug("pass12_entity_created", name=name)
    return entity_id


def run_pass12(
    engine,
    provider: LLMProvider,
    run_id: str,
    data_dir: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """Run Pass 12: IFC schema extraction from source documents.

    Returns summary dict: {chunks_processed, entities_written,
                           assertions_written, constraints_written}.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent.parent / "data"

    pdf_paths = [data_dir / name for name in IFC_SOURCES if (data_dir / name).exists()]
    if not pdf_paths:
        log.warning("pass12_no_sources_found", data_dir=str(data_dir))
        return {"chunks_processed": 0, "entities_written": 0,
                "assertions_written": 0, "constraints_written": 0}

    all_chunks: list[tuple[str, str]] = []
    for pdf_path in pdf_paths:
        chunks = _read_pdf_chunks(pdf_path, PAGES_PER_CHUNK)
        all_chunks.extend(chunks)
        log.info("pass12_pdf_loaded", path=pdf_path.name, chunks=len(chunks))

    log.info("pass12_start", total_chunks=len(all_chunks), model=provider.model_id)

    if dry_run:
        return {
            "chunks_processed": len(all_chunks),
            "entities_written": 0,
            "assertions_written": 0,
            "constraints_written": 0,
        }

    name_cache: dict[str, str] = {}
    # Pre-populate cache from existing ifc_class entities
    with Session(engine) as s:
        for row in s.exec(
            select(EntityRow).where(EntityRow.entity_type == "ifc_class")
        ).all():
            name_cache[row.name.lower()] = row.id

    total_entities = 0
    total_assertions = 0
    total_constraints = 0

    for chunk_id, text in all_chunks:
        progress_key = ("12", chunk_id, provider.model_id)
        with Session(engine) as session:
            existing = session.get(PassProgressRow, progress_key)
            if existing and existing.status == "completed":
                log.debug("pass12_chunk_skip", chunk_id=chunk_id)
                continue

            prompt = CHUNK_PROMPT_TEMPLATE.format(text=text)
            try:
                response: IFCChunkExtractionResponse = provider.extract(
                    prompt, IFCChunkExtractionResponse, entity_name=chunk_id
                )
            except Exception as exc:
                log.warning("pass12_chunk_failed", chunk_id=chunk_id, error=str(exc))
                continue

            now = datetime.now(timezone.utc)
            chunk_entities = 0
            chunk_assertions = 0
            chunk_constraints = 0

            # Write IFC class entities
            for cls in response.ifc_classes:
                if not cls.name.startswith("Ifc"):
                    continue
                _get_or_create_entity(
                    session, cls.name, cls.description,
                    provider, run_id, now, name_cache,
                )
                chunk_entities += 1

            session.flush()

            # Write schema assertions
            for rel in response.schema_relations:
                if not rel.subject_class.startswith("Ifc"):
                    continue
                if not rel.object_class.startswith("Ifc"):
                    continue
                subject_id = _get_or_create_entity(
                    session, rel.subject_class, "",
                    provider, run_id, now, name_cache,
                )
                object_id = _get_or_create_entity(
                    session, rel.object_class, "",
                    provider, run_id, now, name_cache,
                )
                session.flush()

                subject_row = session.get(EntityRow, subject_id)
                object_row = session.get(EntityRow, object_id)
                session.add(AssertionRow(
                    id=str(uuid.uuid4()),
                    subject_id=subject_id,
                    predicate=rel.predicate,
                    object_id=object_id,
                    subject_type=subject_row.entity_type if subject_row else "ifc_class",
                    object_type=object_row.entity_type if object_row else "ifc_class",
                    conditions="[]",
                    exceptions="[]",
                    applicability="[]",
                    source_model=provider.model_id,
                    source_prompt=prompt[:500],
                    created_at=now,
                    extraction_run_id=run_id,
                    confidence=1.0,
                    status="proposed",
                    knowledge_origin="schema",
                    rationale=rel.rationale or None,
                ))
                chunk_assertions += 1

            # Write schema constraints
            for con in response.schema_constraints:
                if not con.subject_class.startswith("Ifc"):
                    continue
                subject_id = _get_or_create_entity(
                    session, con.subject_class, "",
                    provider, run_id, now, name_cache,
                )
                session.flush()

                session.add(ConstraintRow(
                    id=str(uuid.uuid4()),
                    subject_id=subject_id,
                    rule=con.rule,
                    constraint_type=con.constraint_type,
                    conditions="[]",
                    exceptions="[]",
                    source_model=provider.model_id,
                    source_prompt=prompt[:500],
                    created_at=now,
                    extraction_run_id=run_id,
                    confidence=1.0,
                    status="proposed",
                    knowledge_origin="schema",
                    rationale=con.rationale or None,
                ))
                chunk_constraints += 1

            existing_progress = session.get(PassProgressRow, progress_key)
            if existing_progress:
                existing_progress.completed_at = now
                existing_progress.status = "completed"
            else:
                session.add(PassProgressRow(
                    pass_number="12",
                    entity_id=chunk_id,
                    model=provider.model_id,
                    completed_at=now,
                    status="completed",
                ))

            session.commit()
            total_entities += chunk_entities
            total_assertions += chunk_assertions
            total_constraints += chunk_constraints
            log.info(
                "pass12_chunk_done",
                chunk_id=chunk_id,
                entities=chunk_entities,
                assertions=chunk_assertions,
                constraints=chunk_constraints,
            )

    log.info(
        "pass12_complete",
        chunks_processed=len(all_chunks),
        entities_written=total_entities,
        assertions_written=total_assertions,
        constraints_written=total_constraints,
    )
    return {
        "chunks_processed": len(all_chunks),
        "entities_written": total_entities,
        "assertions_written": total_assertions,
        "constraints_written": total_constraints,
    }
