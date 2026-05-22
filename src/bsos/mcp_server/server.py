"""BSOS MCP server — get_requirements and get_dependencies tools.

Each tool call builds its own session from the engine for concurrency safety.
Tool logic is exposed as plain functions so tests can call them directly.
"""
import json as _json

from mcp.server.fastmcp import FastMCP
from sqlmodel import Session, select

from bsos.persistence.database import create_db_engine
from bsos.persistence.models import AssertionRow, EntityAliasRow, EntityRow

REQUIREMENTS_PREDICATES = frozenset({"requires", "depends_on"})


# ---------------------------------------------------------------------------
# Shared helpers (also used directly in tests)
# ---------------------------------------------------------------------------

def resolve_entity(session: Session, name: str) -> EntityRow | None:
    """Case-insensitive entity lookup against name and aliases."""
    name_lower = name.strip().lower()
    for row in session.exec(
        select(EntityRow).where(EntityRow.status != "merged")
    ).all():
        if row.name.lower() == name_lower:
            return row
    for alias_row in session.exec(select(EntityAliasRow)).all():
        if alias_row.alias.lower() == name_lower:
            entity = session.get(EntityRow, alias_row.entity_id)
            if entity and entity.status != "merged":
                return entity
    return None


def _decode_list(json_str: str) -> list[str]:
    try:
        val = _json.loads(json_str)
        return val if isinstance(val, list) else []
    except Exception:
        return []


def _assertion_to_dict(session: Session, row: AssertionRow) -> dict:
    subject = session.get(EntityRow, row.subject_id)
    obj = session.get(EntityRow, row.object_id)
    return {
        "subject": subject.name if subject else row.subject_id,
        "predicate": row.predicate,
        "object": obj.name if obj else row.object_id,
        "confidence": row.confidence,
        "knowledge_origin": row.knowledge_origin,
        "rationale": row.rationale or "",
        "conditions": _decode_list(row.conditions),
        "exceptions": _decode_list(row.exceptions),
        "applicability": _decode_list(row.applicability),
        "cross_prompt_consistency": row.cross_prompt_consistency,
        "status": row.status,
    }


# ---------------------------------------------------------------------------
# Tool implementations (pure session-based, testable without MCP layer)
# ---------------------------------------------------------------------------

def get_requirements_tool(session: Session, entity: str) -> dict:
    """Assertions where entity is subject and predicate is requires or depends_on."""
    entity_row = resolve_entity(session, entity)
    if entity_row is None:
        return {"error": "entity_not_found", "query": entity}

    rows = session.exec(
        select(AssertionRow).where(
            AssertionRow.subject_id == entity_row.id,
            AssertionRow.predicate.in_(list(REQUIREMENTS_PREDICATES)),
        )
    ).all()

    return {
        "entity": entity_row.name,
        "assertions": [_assertion_to_dict(session, r) for r in rows],
    }


def get_dependencies_tool(session: Session, entity: str) -> dict:
    """Assertions where predicate is depends_on and entity is subject or object."""
    entity_row = resolve_entity(session, entity)
    if entity_row is None:
        return {"error": "entity_not_found", "query": entity}

    rows = session.exec(
        select(AssertionRow).where(
            AssertionRow.predicate == "depends_on",
            (AssertionRow.subject_id == entity_row.id)
            | (AssertionRow.object_id == entity_row.id),
        )
    ).all()

    return {
        "entity": entity_row.name,
        "assertions": [_assertion_to_dict(session, r) for r in rows],
    }


# ---------------------------------------------------------------------------
# MCP server factory
# ---------------------------------------------------------------------------

def create_server(db_path: str) -> FastMCP:
    """Return a FastMCP server with get_requirements and get_dependencies tools."""
    mcp = FastMCP("bsos", instructions="BSOS building domain knowledge graph tools.")
    engine = create_db_engine(db_path)

    @mcp.tool(description=get_requirements_tool.__doc__)
    def get_requirements(entity: str) -> dict:
        with Session(engine) as session:
            return get_requirements_tool(session, entity)

    @mcp.tool(description=get_dependencies_tool.__doc__)
    def get_dependencies(entity: str) -> dict:
        with Session(engine) as session:
            return get_dependencies_tool(session, entity)

    return mcp
