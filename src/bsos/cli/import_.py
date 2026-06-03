"""bsos import command — restore knowledge base from a JSON snapshot (as produced by bsos export)."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np

import typer
from sqlmodel import select

app = typer.Typer()

_IMPORT_SOURCE = "imported"
_NOW = datetime.now(timezone.utc)


def _parse_dt(s: str | None) -> datetime:
    if not s:
        return _NOW
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return _NOW


def _enc(v) -> str:
    """Encode a list to JSON text for storage, accepting list or already-encoded str."""
    if isinstance(v, list):
        return json.dumps(v)
    if isinstance(v, str):
        try:
            json.loads(v)
            return v
        except (ValueError, TypeError):
            pass
    return "[]"


def _existing_ids(session, model_cls) -> set[str]:
    return {r.id for r in session.exec(select(model_cls)).all()}


def _build_entity_maps(session) -> tuple[dict[str, str], dict[str, str]]:
    from bsos.persistence.models import EntityRow
    name_to_id: dict[str, str] = {}
    id_to_type: dict[str, str] = {}
    for r in session.exec(select(EntityRow)).all():
        name_to_id[r.name] = r.id
        id_to_type[r.id] = r.entity_type
    return name_to_id, id_to_type


def _resolve(
    ref: str,
    name_to_id: dict[str, str],
    id_to_type: dict[str, str],
) -> tuple[str | None, str | None]:
    """Return (entity_id, entity_type) for a name-or-id reference from the export."""
    if not ref:
        return None, None
    if ref in name_to_id:
        eid = name_to_id[ref]
        return eid, id_to_type.get(eid)
    if ref in id_to_type:
        return ref, id_to_type[ref]
    return None, None


# ---------------------------------------------------------------------------
# Per-type importers — each returns (added, skipped)
# ---------------------------------------------------------------------------

def _import_entities(session, rows: list[dict], replace: bool) -> tuple[int, int]:
    from bsos.persistence.models import EntityRow
    skip_ids = set() if replace else _existing_ids(session, EntityRow)
    added = skipped = 0
    for r in rows:
        eid = r.get("id") or ""
        if not eid:
            skipped += 1
            continue
        if eid in skip_ids:
            skipped += 1
            continue
        session.merge(EntityRow(
            id=eid,
            name=r.get("name", ""),
            entity_type=r.get("entity_type", "component"),
            description=r.get("description", ""),
            status=r.get("status", "proposed"),
            source_model=r.get("source_model", _IMPORT_SOURCE),
            created_at=_parse_dt(r.get("created_at")),
        ))
        added += 1
    return added, skipped


def _import_assertions(
    session,
    rows: list[dict],
    replace: bool,
    name_to_id: dict[str, str],
    id_to_type: dict[str, str],
) -> tuple[int, int]:
    from bsos.persistence.models import AssertionRow
    skip_ids = set() if replace else _existing_ids(session, AssertionRow)
    added = skipped = 0
    for r in rows:
        eid = r.get("id") or ""
        if not eid:
            skipped += 1
            continue
        if eid in skip_ids:
            skipped += 1
            continue
        subject_id, subject_type = _resolve(r.get("subject", ""), name_to_id, id_to_type)
        object_id, object_type = _resolve(r.get("object", ""), name_to_id, id_to_type)
        if not subject_id or not object_id:
            typer.echo(
                f"  warning: skipping assertion {eid} — unresolved entity "
                f"'{r.get('subject')}' or '{r.get('object')}'",
                err=True,
            )
            skipped += 1
            continue
        session.merge(AssertionRow(
            id=eid,
            subject_id=subject_id,
            predicate=r.get("predicate", ""),
            object_id=object_id,
            subject_type=subject_type or "component",
            object_type=object_type or "component",
            conditions=_enc(r.get("conditions", [])),
            exceptions=_enc(r.get("exceptions", [])),
            applicability=_enc(r.get("applicability", [])),
            source_model=_IMPORT_SOURCE,
            created_at=_parse_dt(r.get("created_at")),
            confidence=float(r.get("confidence", 0.5)),
            status=r.get("status", "proposed"),
            knowledge_origin=r.get("knowledge_origin", "imported"),
            rationale=r.get("rationale") or None,
        ))
        added += 1
    return added, skipped


def _import_constraints(
    session,
    rows: list[dict],
    replace: bool,
    name_to_id: dict[str, str],
    id_to_type: dict[str, str],
) -> tuple[int, int]:
    from bsos.persistence.models import ConstraintRow
    skip_ids = set() if replace else _existing_ids(session, ConstraintRow)
    added = skipped = 0
    for r in rows:
        eid = r.get("id") or ""
        if not eid:
            skipped += 1
            continue
        if eid in skip_ids:
            skipped += 1
            continue
        subject_id, _ = _resolve(r.get("subject", ""), name_to_id, id_to_type)
        if not subject_id:
            typer.echo(
                f"  warning: skipping constraint {eid} — unresolved entity '{r.get('subject')}'",
                err=True,
            )
            skipped += 1
            continue
        session.merge(ConstraintRow(
            id=eid,
            subject_id=subject_id,
            rule=r.get("rule", ""),
            constraint_type=r.get("constraint_type", ""),
            conditions=_enc(r.get("conditions", [])),
            exceptions=_enc(r.get("exceptions", [])),
            source_model=_IMPORT_SOURCE,
            created_at=_parse_dt(r.get("created_at")),
            confidence=float(r.get("confidence", 0.5)),
            status=r.get("status", "proposed"),
            knowledge_origin=r.get("knowledge_origin", "imported"),
            rationale=r.get("rationale") or None,
        ))
        added += 1
    return added, skipped


def _import_patterns(
    session,
    rows: list[dict],
    replace: bool,
    name_to_id: dict[str, str],
    id_to_type: dict[str, str],
) -> tuple[int, int]:
    from bsos.persistence.models import PatternRow
    skip_ids = set() if replace else _existing_ids(session, PatternRow)
    added = skipped = 0
    for r in rows:
        eid = r.get("id") or ""
        if not eid:
            skipped += 1
            continue
        if eid in skip_ids:
            skipped += 1
            continue
        subject_ref = r.get("subject", "")
        subject_id = None
        if subject_ref:
            subject_id, _ = _resolve(subject_ref, name_to_id, id_to_type)
        session.merge(PatternRow(
            id=eid,
            name=r.get("name", ""),
            subject_id=subject_id,
            context=_enc(r.get("context", [])),
            problem=r.get("problem", ""),
            solution=r.get("solution", ""),
            consequences=_enc(r.get("consequences", [])),
            emergent_properties=_enc(r.get("emergent_properties", [])),
            source_model=_IMPORT_SOURCE,
            created_at=_parse_dt(r.get("created_at")),
            confidence=float(r.get("confidence", 0.5)),
            status=r.get("status", "proposed"),
            knowledge_origin=r.get("knowledge_origin", "imported"),
            rationale=r.get("rationale") or None,
        ))
        added += 1
    return added, skipped


def _import_forces(session, rows: list[dict], replace: bool) -> tuple[int, int]:
    from bsos.persistence.models import ForceRow
    skip_ids = set() if replace else _existing_ids(session, ForceRow)
    added = skipped = 0
    for r in rows:
        eid = r.get("id") or ""
        if not eid:
            skipped += 1
            continue
        if eid in skip_ids:
            skipped += 1
            continue
        session.merge(ForceRow(
            id=eid,
            name=r.get("name", ""),
            direction=r.get("direction", ""),
            source_model=_IMPORT_SOURCE,
            created_at=_parse_dt(r.get("created_at")),
            confidence=float(r.get("confidence", 0.5)),
            status=r.get("status", "proposed"),
            knowledge_origin=r.get("knowledge_origin", "imported"),
            rationale=r.get("rationale") or None,
        ))
        added += 1
    return added, skipped


def _import_antipatterns(
    session,
    rows: list[dict],
    replace: bool,
    name_to_id: dict[str, str],
    id_to_type: dict[str, str],
) -> tuple[int, int]:
    from bsos.persistence.models import AntiPatternRow
    skip_ids = set() if replace else _existing_ids(session, AntiPatternRow)
    added = skipped = 0
    for r in rows:
        eid = r.get("id") or ""
        if not eid:
            skipped += 1
            continue
        if eid in skip_ids:
            skipped += 1
            continue
        subject_ref = r.get("subject", "")
        subject_id = None
        if subject_ref:
            subject_id, _ = _resolve(subject_ref, name_to_id, id_to_type)
        session.merge(AntiPatternRow(
            id=eid,
            name=r.get("name", ""),
            subject_id=subject_id,
            conditions=_enc(r.get("conditions", [])),
            consequences=_enc(r.get("consequences", [])),
            mitigations=_enc(r.get("mitigations", [])),
            source_model=_IMPORT_SOURCE,
            created_at=_parse_dt(r.get("created_at")),
            confidence=float(r.get("confidence", 0.5)),
            status=r.get("status", "proposed"),
            knowledge_origin=r.get("knowledge_origin", "imported"),
            rationale=r.get("rationale") or None,
        ))
        added += 1
    return added, skipped


def _import_spatial_relations(
    session,
    rows: list[dict],
    replace: bool,
    name_to_id: dict[str, str],
    id_to_type: dict[str, str],
) -> tuple[int, int]:
    from bsos.persistence.models import SpatialRelationRow
    skip_ids = set() if replace else _existing_ids(session, SpatialRelationRow)
    added = skipped = 0
    for r in rows:
        eid = r.get("id") or ""
        if not eid:
            skipped += 1
            continue
        if eid in skip_ids:
            skipped += 1
            continue
        subject_id, _ = _resolve(r.get("subject", ""), name_to_id, id_to_type)
        object_id, _ = _resolve(r.get("object", ""), name_to_id, id_to_type)
        if not subject_id or not object_id:
            typer.echo(
                f"  warning: skipping spatial_relation {eid} — unresolved entity "
                f"'{r.get('subject')}' or '{r.get('object')}'",
                err=True,
            )
            skipped += 1
            continue
        session.merge(SpatialRelationRow(
            id=eid,
            subject_id=subject_id,
            relation=r.get("relation", ""),
            object_id=object_id,
            source_model=_IMPORT_SOURCE,
            created_at=_parse_dt(r.get("created_at")),
            confidence=float(r.get("confidence", 0.5)),
            status=r.get("status", "proposed"),
            knowledge_origin=r.get("knowledge_origin", "imported"),
            rationale=r.get("rationale") or None,
        ))
        added += 1
    return added, skipped


def _import_process_relations(
    session,
    rows: list[dict],
    replace: bool,
    name_to_id: dict[str, str],
    id_to_type: dict[str, str],
) -> tuple[int, int]:
    from bsos.persistence.models import ProcessRelationRow
    skip_ids = set() if replace else _existing_ids(session, ProcessRelationRow)
    added = skipped = 0
    for r in rows:
        eid = r.get("id") or ""
        if not eid:
            skipped += 1
            continue
        if eid in skip_ids:
            skipped += 1
            continue
        predecessor_id, _ = _resolve(r.get("predecessor", ""), name_to_id, id_to_type)
        successor_id, _ = _resolve(r.get("successor", ""), name_to_id, id_to_type)
        if not predecessor_id or not successor_id:
            typer.echo(
                f"  warning: skipping process_relation {eid} — unresolved entity "
                f"'{r.get('predecessor')}' or '{r.get('successor')}'",
                err=True,
            )
            skipped += 1
            continue
        session.merge(ProcessRelationRow(
            id=eid,
            predecessor_id=predecessor_id,
            successor_id=successor_id,
            hard_constraint=bool(r.get("hard_constraint", False)),
            source_model=_IMPORT_SOURCE,
            created_at=_parse_dt(r.get("created_at")),
            confidence=float(r.get("confidence", 0.5)),
            status=r.get("status", "proposed"),
            knowledge_origin=r.get("knowledge_origin", "imported"),
            rationale=r.get("rationale", ""),
        ))
        added += 1
    return added, skipped


def _import_abstraction_nodes(session, rows: list[dict], replace: bool) -> tuple[int, int]:
    from bsos.persistence.models import AbstractionNodeRow
    skip_ids = set() if replace else _existing_ids(session, AbstractionNodeRow)
    added = skipped = 0
    for r in rows:
        eid = r.get("id") or ""
        if not eid:
            skipped += 1
            continue
        if eid in skip_ids:
            skipped += 1
            continue
        # child_ids are not exported (only child_count); restored as empty.
        session.merge(AbstractionNodeRow(
            id=eid,
            statement=r.get("statement", ""),
            child_ids="[]",
            abstraction_rationale=r.get("abstraction_rationale", ""),
            source_model=_IMPORT_SOURCE,
            created_at=_parse_dt(r.get("created_at")),
            confidence=float(r.get("confidence", 0.5)),
            status=r.get("status", "proposed"),
            rationale=r.get("rationale") or None,
        ))
        added += 1
    return added, skipped


# ---------------------------------------------------------------------------
# Embedding index builder
# ---------------------------------------------------------------------------

SEARCH_EMBEDDING_MODEL = "all-mpnet-base-v2"
_EMBED_BATCH_SIZE = 256


def _build_entity_embeddings(session, embedding_model: str = SEARCH_EMBEDDING_MODEL, _embedder=None) -> int:
    """Build EmbeddingRow records for all non-merged entities that lack one.

    _embedder is a test seam; omit in production to use SentenceTransformer.
    Returns the number of embeddings written.
    """
    from bsos.persistence.models import EmbeddingRow, EntityRow

    entities = session.exec(
        select(EntityRow).where(EntityRow.status != "merged")
    ).all()

    to_embed = []
    for entity in entities:
        chash = hashlib.sha256(entity.name.encode()).hexdigest()
        existing = session.get(EmbeddingRow, ("entity", entity.id, embedding_model))
        if existing and existing.content_hash == chash:
            continue
        to_embed.append(entity)

    if not to_embed:
        return 0

    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        model_obj = SentenceTransformer(embedding_model)
        _embedder = model_obj.encode

    added = 0
    for i in range(0, len(to_embed), _EMBED_BATCH_SIZE):
        batch = to_embed[i : i + _EMBED_BATCH_SIZE]
        names = [e.name for e in batch]
        vectors = np.array(_embedder(names), dtype=np.float32)
        for j, entity in enumerate(batch):
            chash = hashlib.sha256(entity.name.encode()).hexdigest()
            vec = vectors[j]
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
            added += 1
        session.commit()

    return added


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def import_cmd(
    input_path: str = typer.Option(..., "--input", "-i", help="JSON file to import, or - for stdin"),
    replace: bool = typer.Option(
        False, "--replace",
        help="Overwrite existing records by ID when the database is non-empty (requires --force)",
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Allow import into a database that already has entities",
    ),
    skip_index: bool = typer.Option(
        False, "--skip-index",
        help="Skip building the entity embedding index after import (search_entities will return empty results)",
    ),
    db: Optional[str] = typer.Option(None, "--db"),
) -> None:
    """Import knowledge base from a JSON snapshot (as produced by bsos export).

    Refuses to import into a non-empty database unless --force is given.
    The LLM response cache is never exported or imported and is always preserved.
    After import, builds entity embeddings for search_entities (use --skip-index to disable).
    """
    from bsos.cli.db_context import open_db
    from bsos.persistence.models import EntityRow

    if input_path == "-":
        try:
            data = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            typer.echo(f"Invalid JSON on stdin: {e}", err=True)
            raise typer.Exit(1)
    else:
        try:
            with open(input_path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            typer.echo(f"File not found: {input_path}", err=True)
            raise typer.Exit(1)
        except json.JSONDecodeError as e:
            typer.echo(f"Invalid JSON in {input_path}: {e}", err=True)
            raise typer.Exit(1)

    if not isinstance(data, dict):
        typer.echo("Expected a JSON object with keys like 'entities', 'assertions', …", err=True)
        raise typer.Exit(1)

    _, session = open_db(db)

    # Guard: refuse to clobber a working database unless --force is explicit.
    with session:
        existing_count = len(session.exec(select(EntityRow)).all())
    if existing_count > 0 and not force:
        typer.echo(
            f"Database already contains {existing_count} entities. "
            "Pass --force to import anyway (existing records are skipped unless "
            "you also pass --replace).",
            err=True,
        )
        raise typer.Exit(1)

    _, session = open_db(db)
    counts: dict[str, tuple[int, int]] = {}

    with session:
        if "entities" in data:
            counts["entities"] = _import_entities(session, data["entities"], replace)
        if "forces" in data:
            counts["forces"] = _import_forces(session, data["forces"], replace)

        # Flush so entity rows are visible to the name-lookup queries below.
        session.flush()
        name_to_id, id_to_type = _build_entity_maps(session)

        if "assertions" in data:
            counts["assertions"] = _import_assertions(
                session, data["assertions"], replace, name_to_id, id_to_type
            )
        if "constraints" in data:
            counts["constraints"] = _import_constraints(
                session, data["constraints"], replace, name_to_id, id_to_type
            )
        if "patterns" in data:
            counts["patterns"] = _import_patterns(
                session, data["patterns"], replace, name_to_id, id_to_type
            )
        if "antipatterns" in data:
            counts["antipatterns"] = _import_antipatterns(
                session, data["antipatterns"], replace, name_to_id, id_to_type
            )
        if "spatial_relations" in data:
            counts["spatial_relations"] = _import_spatial_relations(
                session, data["spatial_relations"], replace, name_to_id, id_to_type
            )
        if "process_relations" in data:
            counts["process_relations"] = _import_process_relations(
                session, data["process_relations"], replace, name_to_id, id_to_type
            )
        if "abstraction_nodes" in data:
            counts["abstraction_nodes"] = _import_abstraction_nodes(
                session, data["abstraction_nodes"], replace
            )

        session.commit()

    total_added = sum(a for a, _ in counts.values())
    total_skipped = sum(s for _, s in counts.values())
    for name, (added, skipped) in counts.items():
        parts = [f"{added} added"]
        if skipped:
            parts.append(f"{skipped} skipped")
        typer.echo(f"  {name}: {', '.join(parts)}")
    typer.echo(f"Total: {total_added} records imported, {total_skipped} skipped")
    if "abstraction_nodes" in counts and counts["abstraction_nodes"][0] > 0:
        typer.echo(
            "  note: abstraction node child_ids are not stored in the export "
            "and were reset to [] — re-run pass 11 to rebuild them.",
            err=True,
        )

    if not skip_index:
        entity_count = counts.get("entities", (0, 0))[0] + counts.get("entities", (0, 0))[1]
        typer.echo(f"Building entity search index… ({entity_count} entities)")
        _, idx_session = open_db(db)
        with idx_session:
            n = _build_entity_embeddings(idx_session)
        typer.echo(f"  embeddings: {n} built")
