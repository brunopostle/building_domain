"""BSOS MCP server — knowledge graph query tools.

Each tool call builds its own session from the engine for concurrency safety.
Tool logic is exposed as plain functions so tests can call them directly.
"""
import json as _json
from collections import deque

import numpy as np
import networkx as nx
from mcp.server.fastmcp import FastMCP
from sqlmodel import Session, select

from bsos.persistence.database import create_db_engine
from bsos.persistence.models import (
    AntiPatternRow, AssertionRow, ConstraintRow, EntityAliasRow, EntityRow,
    EmbeddingRow, ForceRow, PatternRow, ProcessRelationRow, SpatialRelationRow,
)

SEARCH_EMBEDDING_MODEL = "all-mpnet-base-v2"

_embedder_model = None


def _get_embedder():
    global _embedder_model
    if _embedder_model is None:
        from sentence_transformers import SentenceTransformer
        _embedder_model = SentenceTransformer(SEARCH_EMBEDDING_MODEL)
    return _embedder_model
from bsos.graph import build_lazy_subgraph

REQUIREMENTS_PREDICATES = frozenset({"requires", "depends_on"})

_KNOWLEDGE_ORIGIN_ORDER = {"physical": 0, "engineering": 1, "architectural": 2, "cultural": 3}


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


def _apply_shared_params(rows, min_confidence=0.0, include_proposed=True):
    """Filter rows by confidence and proposed status."""
    result = []
    for row in rows:
        if row.confidence < min_confidence:
            continue
        if not include_proposed and row.status == "proposed":
            continue
        result.append(row)
    return result


def _sort_by_confidence_then_origin(rows):
    return sorted(
        rows,
        key=lambda r: (-r.confidence, _KNOWLEDGE_ORIGIN_ORDER.get(r.knowledge_origin, 99)),
    )


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


def get_constraints_tool(
    session: Session,
    entity: str,
    min_confidence: float = 0.0,
    max_results: int = 100,
    include_proposed: bool = True,
) -> dict:
    """Constraint rules where subject_id matches entity."""
    entity_row = resolve_entity(session, entity)
    if entity_row is None:
        return {"error": "entity_not_found", "query": entity}

    rows = session.exec(
        select(ConstraintRow).where(ConstraintRow.subject_id == entity_row.id)
    ).all()
    rows = _apply_shared_params(rows, min_confidence, include_proposed)
    rows = _sort_by_confidence_then_origin(rows)[:max_results]

    return {
        "entity": entity_row.name,
        "constraints": [
            {
                "rule": r.rule,
                "constraint_type": r.constraint_type,
                "conditions": _decode_list(r.conditions),
                "exceptions": _decode_list(r.exceptions),
                "confidence": r.confidence,
                "knowledge_origin": r.knowledge_origin,
                "status": r.status,
            }
            for r in rows
        ],
    }


def get_failure_modes_tool(
    session: Session,
    entity: str,
    min_confidence: float = 0.0,
    max_results: int = 100,
    include_proposed: bool = True,
) -> dict:
    """Anti-pattern failure modes where subject_id matches entity."""
    entity_row = resolve_entity(session, entity)
    if entity_row is None:
        return {"error": "entity_not_found", "query": entity}

    rows = session.exec(
        select(AntiPatternRow).where(AntiPatternRow.subject_id == entity_row.id)
    ).all()
    rows = _apply_shared_params(rows, min_confidence, include_proposed)
    rows = _sort_by_confidence_then_origin(rows)[:max_results]

    return {
        "entity": entity_row.name,
        "failure_modes": [
            {
                "name": r.name,
                "conditions": _decode_list(r.conditions),
                "consequences": _decode_list(r.consequences),
                "mitigations": _decode_list(r.mitigations),
                "confidence": r.confidence,
                "knowledge_origin": r.knowledge_origin,
                "status": r.status,
            }
            for r in rows
        ],
    }


def get_patterns_tool(
    session: Session,
    entity: str,
    min_confidence: float = 0.0,
    max_results: int = 100,
    include_proposed: bool = True,
) -> dict:
    """Architectural patterns where subject_id matches entity."""
    entity_row = resolve_entity(session, entity)
    if entity_row is None:
        return {"error": "entity_not_found", "query": entity}

    rows = session.exec(
        select(PatternRow).where(PatternRow.subject_id == entity_row.id)
    ).all()
    rows = _apply_shared_params(rows, min_confidence, include_proposed)
    rows = _sort_by_confidence_then_origin(rows)[:max_results]

    result_patterns = []
    for r in rows:
        force_ids = _decode_list(r.force_ids)
        force_descriptions = _decode_list(r.force_descriptions)
        forces_warning = None

        if force_ids:
            forces = []
            for fid in force_ids:
                frow = session.get(ForceRow, fid)
                forces.append(frow.name if frow else fid)
        elif force_descriptions:
            forces = force_descriptions
            forces_warning = "force_ids not yet resolved; using raw descriptions"
        else:
            forces = []

        entry = {
            "name": r.name,
            "problem": r.problem,
            "solution": r.solution,
            "context": _decode_list(r.context),
            "forces": forces,
            "consequences": _decode_list(r.consequences),
            "emergent_properties": _decode_list(r.emergent_properties),
            "confidence": r.confidence,
            "knowledge_origin": r.knowledge_origin,
            "status": r.status,
        }
        if forces_warning:
            entry["forces_warning"] = forces_warning
        result_patterns.append(entry)

    return {"entity": entity_row.name, "patterns": result_patterns}


def get_forces_tool(
    session: Session,
    entity: str,
    min_confidence: float = 0.0,
    max_results: int = 100,
    include_proposed: bool = True,
) -> dict:
    """Forces (design pressures) where entity UUID appears in affects JSON array."""
    entity_row = resolve_entity(session, entity)
    if entity_row is None:
        return {"error": "entity_not_found", "query": entity}

    all_forces = session.exec(select(ForceRow)).all()
    matching = [
        r for r in all_forces
        if entity_row.id in _decode_list(r.affects)
    ]
    matching = _apply_shared_params(matching, min_confidence, include_proposed)
    matching = _sort_by_confidence_then_origin(matching)[:max_results]

    return {
        "entity": entity_row.name,
        "forces": [
            {
                "name": r.name,
                "direction": r.direction,
                "confidence": r.confidence,
                "knowledge_origin": r.knowledge_origin,
                "rationale": r.rationale or "",
                "status": r.status,
            }
            for r in matching
        ],
    }


def get_spatial_relations_tool(
    session: Session,
    entity: str,
    min_confidence: float = 0.0,
    max_results: int = 100,
    include_proposed: bool = True,
) -> dict:
    """Spatial relations where entity is subject or object."""
    entity_row = resolve_entity(session, entity)
    if entity_row is None:
        return {"error": "entity_not_found", "query": entity}

    rows = session.exec(
        select(SpatialRelationRow).where(
            (SpatialRelationRow.subject_id == entity_row.id)
            | (SpatialRelationRow.object_id == entity_row.id)
        )
    ).all()
    rows = _apply_shared_params(rows, min_confidence, include_proposed)
    rows = _sort_by_confidence_then_origin(rows)[:max_results]

    def _name(eid: str) -> str:
        e = session.get(EntityRow, eid)
        return e.name if e else eid

    return {
        "entity": entity_row.name,
        "spatial_relations": [
            {
                "subject": _name(r.subject_id),
                "relation": r.relation,
                "object": _name(r.object_id),
                "confidence": r.confidence,
                "knowledge_origin": r.knowledge_origin,
                "status": r.status,
            }
            for r in rows
        ],
    }


def get_process_sequence_tool(
    session: Session,
    entity: str,
    max_depth: int = 50,
) -> dict:
    """Process sequence subgraph reachable from entity in either direction.

    Builds a per-request lazy subgraph from the graph layer (no shared state).
    """
    entity_row = resolve_entity(session, entity)
    if entity_row is None:
        return {"error": "entity_not_found", "query": entity}

    sub = build_lazy_subgraph(session, entity_row.id)

    def _name(eid: str) -> str:
        node = sub.nodes.get(eid)
        if node:
            return node.get("name", eid)
        e = session.get(EntityRow, eid)
        return e.name if e else eid

    # Extract process-only graph (precedes edges between entity nodes)
    g = nx.DiGraph(
        (u, v) for u, v, d in sub.edges(data=True)
        if d.get("edge_type") == "precedes"
    )

    start = entity_row.id
    if start not in g:
        return {
            "entity": entity_row.name,
            "sequence": [entity_row.name],
            "has_cycle": False,
            "truncated": False,
        }

    # BFS reachable in both directions within max_depth
    reachable: set[str] = {start}
    frontier = deque([(start, 0)])
    truncated = False
    while frontier:
        node, depth = frontier.popleft()
        if depth >= max_depth:
            truncated = True
            continue
        for nb in list(g.predecessors(node)) + list(g.successors(node)):
            if nb not in reachable:
                reachable.add(nb)
                frontier.append((nb, depth + 1))

    subgraph = g.subgraph(reachable)

    if not nx.is_directed_acyclic_graph(subgraph):
        try:
            cycle = nx.find_cycle(subgraph)
            cycle_desc = " -> ".join(_name(u) for u, v in cycle) + f" -> {_name(cycle[0][0])}"
        except Exception:
            cycle_desc = "cycle detected"
        return {
            "entity": entity_row.name,
            "sequence": [_name(n) for n in reachable],
            "has_cycle": True,
            "cycle_description": cycle_desc,
            "truncated": truncated,
        }

    sequence = [_name(n) for n in nx.topological_sort(subgraph)]
    return {
        "entity": entity_row.name,
        "sequence": sequence,
        "has_cycle": False,
        "truncated": truncated,
    }


def search_entities_tool(
    session: Session,
    query: str,
    max_results: int = 10,
    min_score: float = 0.0,
    _embedder=None,
) -> dict:
    """Semantic search over bsos entities using embedding similarity.

    Accepts a free-text description or IFC element name and returns ranked
    matching entities. Use this before other tools when you don't know the
    exact bsos entity name.
    """
    embedder = _embedder or _get_embedder()
    query_vec = np.array(embedder.encode([query])[0], dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return {"error": "empty_query", "query": query, "results": []}

    rows = session.exec(
        select(EmbeddingRow).where(
            EmbeddingRow.item_type == "entity",
            EmbeddingRow.model == SEARCH_EMBEDDING_MODEL,
        )
    ).all()

    scores = []
    for row in rows:
        vec = np.frombuffer(row.vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        score = float(np.dot(query_vec, vec) / (query_norm * norm))
        if score >= min_score:
            scores.append((score, row.item_id))

    scores.sort(key=lambda x: -x[0])
    top = scores[:max_results]

    results = []
    for score, entity_id in top:
        entity = session.get(EntityRow, entity_id)
        if entity is None or entity.status == "merged":
            continue
        results.append({
            "name": entity.name,
            "score": round(score, 4),
            "entity_type": entity.entity_type,
            "description": entity.description,
        })

    return {"query": query, "results": results}


# ---------------------------------------------------------------------------
# MCP server factory
# ---------------------------------------------------------------------------

def create_server(db_path: str) -> FastMCP:
    """Return a FastMCP server with all BSOS knowledge graph query tools."""
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

    @mcp.tool(description=get_constraints_tool.__doc__)
    def get_constraints(
        entity: str,
        min_confidence: float = 0.0,
        max_results: int = 100,
        include_proposed: bool = True,
    ) -> dict:
        with Session(engine) as session:
            return get_constraints_tool(session, entity, min_confidence, max_results, include_proposed)

    @mcp.tool(description=get_failure_modes_tool.__doc__)
    def get_failure_modes(
        entity: str,
        min_confidence: float = 0.0,
        max_results: int = 100,
        include_proposed: bool = True,
    ) -> dict:
        with Session(engine) as session:
            return get_failure_modes_tool(session, entity, min_confidence, max_results, include_proposed)

    @mcp.tool(description=get_patterns_tool.__doc__)
    def get_patterns(
        entity: str,
        min_confidence: float = 0.0,
        max_results: int = 100,
        include_proposed: bool = True,
    ) -> dict:
        with Session(engine) as session:
            return get_patterns_tool(session, entity, min_confidence, max_results, include_proposed)

    @mcp.tool(description=get_forces_tool.__doc__)
    def get_forces(
        entity: str,
        min_confidence: float = 0.0,
        max_results: int = 100,
        include_proposed: bool = True,
    ) -> dict:
        with Session(engine) as session:
            return get_forces_tool(session, entity, min_confidence, max_results, include_proposed)

    @mcp.tool(description=get_spatial_relations_tool.__doc__)
    def get_spatial_relations(
        entity: str,
        min_confidence: float = 0.0,
        max_results: int = 100,
        include_proposed: bool = True,
    ) -> dict:
        with Session(engine) as session:
            return get_spatial_relations_tool(session, entity, min_confidence, max_results, include_proposed)

    @mcp.tool(description=get_process_sequence_tool.__doc__)
    def get_process_sequence(entity: str, max_depth: int = 50) -> dict:
        with Session(engine) as session:
            return get_process_sequence_tool(session, entity, max_depth)

    @mcp.tool(description=search_entities_tool.__doc__)
    def search_entities(
        query: str,
        max_results: int = 10,
        min_score: float = 0.0,
    ) -> dict:
        with Session(engine) as session:
            return search_entities_tool(session, query, max_results, min_score)

    return mcp
