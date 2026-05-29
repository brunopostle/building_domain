"""bsos export command — export knowledge base to JSON or CSV."""
import csv
import io
import json
import sys
from typing import Optional

import typer
from sqlmodel import select

app = typer.Typer()

ALL_TYPES = [
    "entities",
    "assertions",
    "constraints",
    "patterns",
    "forces",
    "antipatterns",
    "spatial_relations",
    "process_relations",
    "abstraction_nodes",
]


def _entity_map(session) -> dict[str, str]:
    """Return id → name mapping for all non-merged entities."""
    from bsos.persistence.models import EntityRow
    return {
        r.id: r.name
        for r in session.exec(select(EntityRow).where(EntityRow.status != "merged")).all()
    }


def _decode(json_str: str) -> list:
    try:
        v = json.loads(json_str)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def _export_entities(session, status_filter) -> list[dict]:
    from bsos.persistence.models import EntityRow
    rows = session.exec(select(EntityRow)).all()
    result = []
    for r in rows:
        if status_filter and r.status not in status_filter:
            continue
        result.append({
            "id": r.id,
            "name": r.name,
            "entity_type": r.entity_type,
            "description": r.description or "",
            "status": r.status,
            "source_model": r.source_model,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })
    return result


def _export_assertions(session, status_filter, names) -> list[dict]:
    from bsos.persistence.models import AssertionRow
    rows = session.exec(select(AssertionRow)).all()
    result = []
    for r in rows:
        if status_filter and r.status not in status_filter:
            continue
        result.append({
            "id": r.id,
            "subject": names.get(r.subject_id, r.subject_id),
            "predicate": r.predicate,
            "object": names.get(r.object_id, r.object_id),
            "confidence": r.confidence,
            "knowledge_origin": r.knowledge_origin,
            "rationale": r.rationale or "",
            "applicability": _decode(r.applicability),
            "conditions": _decode(r.conditions),
            "exceptions": _decode(r.exceptions),
            "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })
    return result


def _export_constraints(session, status_filter, names) -> list[dict]:
    from bsos.persistence.models import ConstraintRow
    rows = session.exec(select(ConstraintRow)).all()
    result = []
    for r in rows:
        if status_filter and r.status not in status_filter:
            continue
        result.append({
            "id": r.id,
            "subject": names.get(r.subject_id, r.subject_id),
            "rule": r.rule,
            "constraint_type": r.constraint_type,
            "conditions": _decode(r.conditions),
            "exceptions": _decode(r.exceptions),
            "confidence": r.confidence,
            "knowledge_origin": r.knowledge_origin,
            "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })
    return result


def _export_patterns(session, status_filter, names) -> list[dict]:
    from bsos.persistence.models import PatternRow
    rows = session.exec(select(PatternRow)).all()
    result = []
    for r in rows:
        if status_filter and r.status not in status_filter:
            continue
        result.append({
            "id": r.id,
            "name": r.name,
            "subject": names.get(r.subject_id, r.subject_id) if r.subject_id else "",
            "context": _decode(r.context),
            "problem": r.problem,
            "solution": r.solution,
            "consequences": _decode(r.consequences),
            "emergent_properties": _decode(r.emergent_properties),
            "confidence": r.confidence,
            "knowledge_origin": r.knowledge_origin,
            "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })
    return result


def _export_forces(session, status_filter) -> list[dict]:
    from bsos.persistence.models import ForceRow
    rows = session.exec(select(ForceRow)).all()
    result = []
    for r in rows:
        if status_filter and r.status not in status_filter:
            continue
        result.append({
            "id": r.id,
            "name": r.name,
            "direction": r.direction,
            "rationale": r.rationale or "",
            "confidence": r.confidence,
            "knowledge_origin": r.knowledge_origin,
            "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })
    return result


def _export_antipatterns(session, status_filter, names) -> list[dict]:
    from bsos.persistence.models import AntiPatternRow
    rows = session.exec(select(AntiPatternRow)).all()
    result = []
    for r in rows:
        if status_filter and r.status not in status_filter:
            continue
        result.append({
            "id": r.id,
            "name": r.name,
            "subject": names.get(r.subject_id, r.subject_id) if r.subject_id else "",
            "conditions": _decode(r.conditions),
            "consequences": _decode(r.consequences),
            "mitigations": _decode(r.mitigations),
            "confidence": r.confidence,
            "knowledge_origin": r.knowledge_origin,
            "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })
    return result


def _export_spatial_relations(session, status_filter, names) -> list[dict]:
    from bsos.persistence.models import SpatialRelationRow
    rows = session.exec(select(SpatialRelationRow)).all()
    result = []
    for r in rows:
        if status_filter and r.status not in status_filter:
            continue
        result.append({
            "id": r.id,
            "subject": names.get(r.subject_id, r.subject_id),
            "relation": r.relation,
            "object": names.get(r.object_id, r.object_id),
            "confidence": r.confidence,
            "knowledge_origin": r.knowledge_origin,
            "rationale": r.rationale or "",
            "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })
    return result


def _export_process_relations(session, status_filter, names) -> list[dict]:
    from bsos.persistence.models import ProcessRelationRow
    rows = session.exec(select(ProcessRelationRow)).all()
    result = []
    for r in rows:
        if status_filter and r.status not in status_filter:
            continue
        result.append({
            "id": r.id,
            "predecessor": names.get(r.predecessor_id, r.predecessor_id),
            "successor": names.get(r.successor_id, r.successor_id),
            "hard_constraint": r.hard_constraint,
            "rationale": r.rationale or "",
            "confidence": r.confidence,
            "knowledge_origin": r.knowledge_origin,
            "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })
    return result


def _export_abstraction_nodes(session, status_filter) -> list[dict]:
    from bsos.persistence.models import AbstractionNodeRow
    rows = session.exec(select(AbstractionNodeRow)).all()
    result = []
    for r in rows:
        if status_filter and r.status not in status_filter:
            continue
        result.append({
            "id": r.id,
            "statement": r.statement,
            "child_count": len(_decode(r.child_ids)),
            "abstraction_rationale": r.abstraction_rationale,
            "confidence": r.confidence,
            "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })
    return result


def _gather(session, types: list[str], status_filter: set | None) -> dict[str, list[dict]]:
    names = _entity_map(session)
    result = {}
    dispatch = {
        "entities": lambda: _export_entities(session, status_filter),
        "assertions": lambda: _export_assertions(session, status_filter, names),
        "constraints": lambda: _export_constraints(session, status_filter, names),
        "patterns": lambda: _export_patterns(session, status_filter, names),
        "forces": lambda: _export_forces(session, status_filter),
        "antipatterns": lambda: _export_antipatterns(session, status_filter, names),
        "spatial_relations": lambda: _export_spatial_relations(session, status_filter, names),
        "process_relations": lambda: _export_process_relations(session, status_filter, names),
        "abstraction_nodes": lambda: _export_abstraction_nodes(session, status_filter),
    }
    for t in types:
        result[t] = dispatch[t]()
    return result


def _write_csv(rows: list[dict], dest) -> None:
    if not rows:
        return
    writer = csv.DictWriter(dest, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for row in rows:
        # Serialize list fields as JSON strings for CSV compatibility
        flat = {
            k: json.dumps(v) if isinstance(v, list) else v
            for k, v in row.items()
        }
        writer.writerow(flat)


@app.callback(invoke_without_command=True)
def export_cmd(
    fmt: str = typer.Option("json", "--format", "-f", help="Output format: json or csv"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output path (default: stdout)"),
    item_types: Optional[list[str]] = typer.Option(
        None, "--type", "-t",
        help="Item type(s) to export. Repeat for multiple. Defaults to all.",
    ),
    status: Optional[list[str]] = typer.Option(
        None, "--status", "-s",
        help="Status filter (proposed, accepted, deprecated). Repeat for multiple. Defaults to all.",
    ),
    db: Optional[str] = typer.Option(None, "--db"),
) -> None:
    """Export the knowledge base to JSON or CSV."""
    from bsos.cli.db_context import open_db

    fmt = fmt.lower()
    if fmt not in ("json", "csv"):
        typer.echo(f"Unknown format '{fmt}'. Use json or csv.", err=True)
        raise typer.Exit(1)

    types = item_types if item_types else ALL_TYPES
    for t in types:
        if t not in ALL_TYPES:
            typer.echo(f"Unknown type '{t}'. Choose from: {', '.join(ALL_TYPES)}", err=True)
            raise typer.Exit(1)

    if fmt == "csv" and len(types) > 1:
        typer.echo(
            "CSV format supports one --type at a time. "
            "Specify e.g. --type entities, or use --format json for a multi-type export.",
            err=True,
        )
        raise typer.Exit(1)

    status_filter = set(status) if status else None

    _, session = open_db(db)
    with session:
        data = _gather(session, types, status_filter)

    if fmt == "json":
        text = json.dumps(data, indent=2, ensure_ascii=False)
        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(text)
            total = sum(len(v) for v in data.values())
            typer.echo(f"Exported {total} records to {output}")
        else:
            typer.echo(text)
    else:
        # CSV — single type guaranteed above
        type_name = types[0]
        rows = data[type_name]
        if output:
            with open(output, "w", newline="", encoding="utf-8") as f:
                _write_csv(rows, f)
            typer.echo(f"Exported {len(rows)} {type_name} records to {output}")
        else:
            buf = io.StringIO()
            _write_csv(rows, buf)
            sys.stdout.write(buf.getvalue())
