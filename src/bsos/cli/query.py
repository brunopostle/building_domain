"""bsos query command."""
import json
from sqlmodel import select
from bsos.persistence.models import EntityRow, AssertionRow, EntityAliasRow

ORIGIN_PRIORITY = {"physical": 0, "engineering": 1, "architectural": 2, "cultural": 3}


def resolve_entity(session, name: str) -> EntityRow | None:
    row = session.exec(
        select(EntityRow).where(EntityRow.name.ilike(name))  # type: ignore[attr-defined]
    ).first()
    if row:
        return row
    alias = session.exec(
        select(EntityAliasRow).where(EntityAliasRow.alias.ilike(name))  # type: ignore[attr-defined]
    ).first()
    if alias:
        return session.get(EntityRow, alias.entity_id)
    return None


def _entity_name(session, entity_id: str) -> str:
    row = session.get(EntityRow, entity_id)
    return row.name if row else entity_id


def get_assertions(session, entity_row: EntityRow, min_confidence: float, include_proposed: bool) -> list[dict]:
    statuses = ["accepted"]
    if include_proposed:
        statuses.append("proposed")

    rows = session.exec(
        select(AssertionRow).where(
            (AssertionRow.subject_id == entity_row.id) | (AssertionRow.object_id == entity_row.id),
            AssertionRow.status.in_(statuses),  # type: ignore[attr-defined]
            AssertionRow.confidence >= min_confidence,
        )
    ).all()

    results = []
    for row in rows:
        results.append({
            "subject": _entity_name(session, row.subject_id),
            "predicate": row.predicate,
            "object": _entity_name(session, row.object_id),
            "confidence": row.confidence,
            "knowledge_origin": row.knowledge_origin,
            "status": row.status,
            "conditions": json.loads(row.conditions) if row.conditions else [],
            "exceptions": json.loads(row.exceptions) if row.exceptions else [],
            "applicability": json.loads(row.applicability) if row.applicability else [],
            "cross_prompt_consistency": row.cross_prompt_consistency,
        })

    results.sort(key=lambda r: (
        -r["confidence"],
        ORIGIN_PRIORITY.get(r["knowledge_origin"], 99),
    ))
    return results


def format_assertions(results: list[dict], entity: str) -> str:
    if not results:
        return f"No assertions found for '{entity}'."
    lines = [f"Assertions for '{entity}'  ({len(results)} results)\n"]
    for r in results:
        conf = f"{r['confidence']:.2f}"
        lines.append(f"  {r['subject']} —[{r['predicate']}]→ {r['object']}")
        lines.append(f"    confidence={conf}  origin={r['knowledge_origin']}  status={r['status']}")
        if r["conditions"]:
            lines.append(f"    conditions: {'; '.join(r['conditions'])}")
        if r["exceptions"]:
            lines.append(f"    exceptions: {'; '.join(r['exceptions'])}")
        lines.append("")
    return "\n".join(lines)


def format_constraints(result: dict) -> str:
    items = result.get("constraints", [])
    entity = result.get("entity", "?")
    if not items:
        return f"No constraints found for '{entity}'."
    lines = [f"Constraints for '{entity}'  ({len(items)} results)\n"]
    for c in items:
        lines.append(f"  [{c['constraint_type'].upper()}] {c['rule']}")
        lines.append(f"    confidence={c['confidence']:.2f}  origin={c['knowledge_origin']}  status={c['status']}")
        if c.get("conditions"):
            lines.append(f"    conditions: {'; '.join(c['conditions'])}")
        if c.get("exceptions"):
            lines.append(f"    exceptions: {'; '.join(c['exceptions'])}")
        lines.append("")
    return "\n".join(lines)


def format_failure_modes(result: dict) -> str:
    items = result.get("failure_modes", [])
    entity = result.get("entity", "?")
    if not items:
        return f"No failure modes found for '{entity}'."
    lines = [f"Failure modes for '{entity}'  ({len(items)} results)\n"]
    for fm in items:
        lines.append(f"  {fm['name']}")
        lines.append(f"    confidence={fm['confidence']:.2f}  origin={fm['knowledge_origin']}  status={fm['status']}")
        if fm.get("conditions"):
            lines.append(f"    conditions: {'; '.join(fm['conditions'])}")
        if fm.get("consequences"):
            lines.append(f"    consequences: {'; '.join(fm['consequences'])}")
        if fm.get("mitigations"):
            lines.append(f"    mitigations: {'; '.join(fm['mitigations'])}")
        lines.append("")
    return "\n".join(lines)


def format_forces(result: dict) -> str:
    items = result.get("forces", [])
    entity = result.get("entity", "?")
    if not items:
        return f"No forces found affecting '{entity}'."
    lines = [f"Forces affecting '{entity}'  ({len(items)} results)\n"]
    for f in items:
        lines.append(f"  [{f['direction'].upper()}] {f['name']}")
        lines.append(f"    confidence={f['confidence']:.2f}  origin={f['knowledge_origin']}  status={f['status']}")
        if f.get("rationale"):
            lines.append(f"    rationale: {f['rationale']}")
        lines.append("")
    return "\n".join(lines)


def format_spatial_relations(result: dict) -> str:
    items = result.get("spatial_relations", [])
    entity = result.get("entity", "?")
    if not items:
        return f"No spatial relations found for '{entity}'."
    lines = [f"Spatial relations for '{entity}'  ({len(items)} results)\n"]
    for sr in items:
        lines.append(f"  {sr['subject']} —[{sr['relation']}]→ {sr['object']}")
        lines.append(f"    confidence={sr['confidence']:.2f}  origin={sr['knowledge_origin']}  status={sr['status']}")
        lines.append("")
    return "\n".join(lines)


def format_process_sequence(result: dict) -> str:
    entity = result.get("entity", "?")
    if "error" in result:
        return f"Entity '{entity}' not found."
    seq = result.get("sequence", [])
    has_cycle = result.get("has_cycle", False)
    truncated = result.get("truncated", False)
    lines = [f"Process sequence for '{entity}'\n"]
    if has_cycle:
        lines.append(f"  WARNING: cycle detected — {result.get('cycle_description', '')}")
        lines.append("")
    for i, name in enumerate(seq, 1):
        marker = "→" if name != entity else "●"
        lines.append(f"  {i:3}. {marker} {name}")
    if truncated:
        lines.append("  ... (truncated at max_depth)")
    return "\n".join(lines)
