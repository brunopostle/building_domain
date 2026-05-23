"""NetworkX graph construction layer (Section 11).

Two build modes:
  build_full_graph     — complete graph, used by bsos build-graph and serialization
  build_lazy_subgraph  — BFS subgraph from one entity, for per-request CLI/MCP use

Node types: entity, assertion, abstraction_node, pattern, force, antipattern
Edge categories: assertion, structural, spatial
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path

import joblib
import networkx as nx
from sqlmodel import Session, select

from bsos.persistence.models import (
    AbstractionNodeRow,
    AntiPatternRow,
    AssertionRow,
    EntityRow,
    ForceRow,
    PatternRow,
    ProcessRelationRow,
    SpatialRelationRow,
)


def _status_filter(min_status: str) -> set[str]:
    if min_status == "accepted":
        return {"accepted"}
    return {"accepted", "proposed"}


def _decode_list(json_str: str) -> list[str]:
    try:
        val = json.loads(json_str)
        return val if isinstance(val, list) else []
    except Exception:
        return []


def build_full_graph(session: Session, min_status: str = "proposed") -> nx.DiGraph:
    """Build complete directed graph from all SQLite records."""
    allowed = _status_filter(min_status)
    g = nx.DiGraph()

    for row in session.exec(select(EntityRow)).all():
        g.add_node(row.id, node_type="entity", name=row.name,
                   entity_type=row.entity_type, status=row.status)

    for row in session.exec(select(AssertionRow)).all():
        if row.status not in allowed:
            continue
        g.add_node(row.id, node_type="assertion", predicate=row.predicate,
                   subject_id=row.subject_id, object_id=row.object_id, status=row.status)
        if row.subject_id in g and row.object_id in g:
            g.add_edge(row.subject_id, row.object_id,
                       edge_type=row.predicate, assertion_id=row.id,
                       edge_category="assertion")

    for row in session.exec(select(AbstractionNodeRow)).all():
        if row.status not in allowed:
            continue
        g.add_node(row.id, node_type="abstraction_node",
                   statement=row.statement, status=row.status)
        for child_id in _decode_list(row.child_ids):
            g.add_edge(row.id, child_id, edge_type="aggregates", edge_category="structural")

    for row in session.exec(select(PatternRow)).all():
        if row.status not in allowed:
            continue
        g.add_node(row.id, node_type="pattern", name=row.name, status=row.status)
        if row.subject_id and row.subject_id in g:
            g.add_edge(row.id, row.subject_id, edge_type="applies_to", edge_category="structural")

    for row in session.exec(select(ForceRow)).all():
        if row.status not in allowed:
            continue
        g.add_node(row.id, node_type="force", name=row.name,
                   direction=row.direction, status=row.status)
        for entity_id in _decode_list(row.affects):
            if entity_id in g:
                g.add_edge(row.id, entity_id, edge_type="acts_on", edge_category="structural")

    for row in session.exec(select(AntiPatternRow)).all():
        if row.status not in allowed:
            continue
        g.add_node(row.id, node_type="antipattern", name=row.name, status=row.status)
        if row.subject_id and row.subject_id in g:
            g.add_edge(row.id, row.subject_id, edge_type="applies_to", edge_category="structural")

    for row in session.exec(select(ProcessRelationRow)).all():
        if row.status not in allowed:
            continue
        if row.predecessor_id in g and row.successor_id in g:
            g.add_edge(row.predecessor_id, row.successor_id,
                       edge_type="precedes", edge_category="structural",
                       hard_constraint=row.hard_constraint)

    for row in session.exec(select(SpatialRelationRow)).all():
        if row.status not in allowed:
            continue
        if row.subject_id in g and row.object_id in g:
            g.add_edge(row.subject_id, row.object_id,
                       edge_type=row.relation, edge_category="spatial")

    return g


def build_lazy_subgraph(session: Session, entity_id: str, min_status: str = "proposed") -> nx.DiGraph:
    """Build subgraph of nodes reachable from entity_id (BFS, undirected).

    Each call builds its own local graph — no shared state between requests.
    """
    full = build_full_graph(session, min_status)
    if entity_id not in full:
        return nx.DiGraph()
    reachable = nx.bfs_tree(full.to_undirected(as_view=True), entity_id).nodes()
    return full.subgraph(reachable).copy()


def get_schema_version(engine) -> str:
    """Return current Alembic revision head, or 'unknown'."""
    try:
        from alembic.runtime.migration import MigrationContext
        with engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            return ctx.get_current_revision() or "unknown"
    except Exception:
        return "unknown"


def save_graph(g: nx.DiGraph, path: str | Path, schema_version: str) -> None:
    """Serialize graph to path with SHA-256 companion file. Writes atomically."""
    path = Path(path)
    sha_path = Path(str(path) + ".sha256")
    dir_ = path.parent
    payload = {"graph": g, "schema_version": schema_version}

    with tempfile.NamedTemporaryFile(dir=dir_, delete=False, suffix=".tmp") as f:
        tmp_pkl = Path(f.name)
    joblib.dump(payload, tmp_pkl, compress=3)

    checksum = hashlib.sha256(tmp_pkl.read_bytes()).hexdigest()

    with tempfile.NamedTemporaryFile(dir=dir_, delete=False, suffix=".tmp", mode="w") as f:
        tmp_sha = Path(f.name)
        f.write(checksum)

    os.replace(tmp_pkl, path)
    os.replace(tmp_sha, sha_path)


def load_graph(path: str | Path, engine) -> nx.DiGraph:
    """Load graph from disk, verifying SHA-256 and schema version.

    Raises ValueError if the checksum or schema version does not match.
    """
    path = Path(path)
    sha_path = Path(str(path) + ".sha256")

    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    expected = sha_path.read_text().strip()
    if actual != expected:
        raise ValueError(f"Graph checksum mismatch for {path}")

    payload = joblib.load(path)
    saved_version = payload.get("schema_version", "unknown")
    live_version = get_schema_version(engine)
    if saved_version != live_version:
        raise ValueError(
            f"Schema version mismatch: saved={saved_version!r}, live={live_version!r}. "
            "Run 'bsos build-graph' to rebuild."
        )
    return payload["graph"]
