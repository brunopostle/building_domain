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
