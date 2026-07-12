#!/usr/bin/env python3
"""IFC × BSOS compliance report.

For every space in an IFC model this checks four BSOS knowledge layers
against what is actually modelled:

  * REQUIREMENTS  — requires/depends_on assertions (materials, elements, MEP)
  * CONSTRAINTS   — `must` rules from the constraints table
  * SPATIAL       — adjacent_to/connects_to/accessible_from relations vs the
                    IFC spatial structure (shared bounding walls)
  * ANTIPATTERNS  — known failure conditions, flagged only when the model
                    affirmatively exhibits the negative signal

Each modelled IfcSpace is resolved to a bsos 'space' entity via semantic
search (the same embedding similarity the `search_entities` MCP tool uses),
so the report works against any space in any model rather than a fixed list
of 6 hardcoded space types (building_domain-l5w.1).

Usage:
    python scripts/ifc_compliance_report.py [path/to/model.ifc] [path/to/bsos.db]
"""
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element
import ifcopenshell.util.shape
import numpy as np
from sqlmodel import Session

# The deterministic check engine is shared with the bsos `validate_element` MCP
# tool, so the report and the tool can never disagree. This script's job is to
# extract `facts` from the IFC model and hand them to that engine.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bsos import validation  # noqa: E402
from bsos.persistence.database import create_db_engine  # noqa: E402
from bsos.mcp_server.server import search_entities_tool, resolve_entity, _get_embedder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IFC_PATH = ROOT / "_test.ifc"
BSOS_DB  = ROOT / "bsos.db"

# ── Space semantic matching threshold ────────────────────────────────────────
# Calibrated against production bsos.db (2026-07-02): a genuine EPset_Topology
# usage tag stripped of its numeric suffix (e.g. 'living-2' -> 'living') scores
# >=0.51 against its correct bsos space entity; unrelated text (e.g. a stray
# 'nonsense-zone-9' tag) scores ~0.26. 0.4 separates the two with margin.
SPACE_MATCH_MIN_SCORE = 0.4

# ── IFC element types that represent each system ─────────────────────────────
# Keys must match validation.SYSTEM_OBJECTS (the shared system vocabulary); this
# is the ifc-side mapping of each system name to the classes that evidence it.
MEP_PRESENCE: dict[str, list[str]] = {
    "Ventilation System":     ["IfcAirTerminal", "IfcDuctSegment", "IfcDuctFitting",
                               "IfcFan", "IfcUnitaryEquipment"],
    "HVAC System":            ["IfcAirTerminal", "IfcDuctSegment", "IfcCoil",
                               "IfcUnitaryEquipment", "IfcSpaceHeater"],
    "Heating System":         ["IfcSpaceHeater", "IfcUnitaryEquipment", "IfcBoiler"],
    "Electrical System":      ["IfcElectricDistributionBoard", "IfcLightFixture",
                               "IfcElectricAppliance", "IfcProtectiveDevice",
                               "IfcSwitchingDevice", "IfcOutlet"],
    "Lighting System":        ["IfcLightFixture"],
    # Fixtures (not pipe segments) are the reliable proxy — pipe segments can be
    # rainwater gutters, which are unrelated to interior drainage.
    "Drainage System":        ["IfcSanitaryTerminal", "IfcWasteTerminal"],
    "Rough-in Plumbing":      ["IfcSanitaryTerminal", "IfcValve"],
    "Plumbing System":        ["IfcSanitaryTerminal", "IfcValve"],
    "Fire Suppression System":["IfcFireSuppressionTerminal"],
    "Fire Protection System": ["IfcFireSuppressionTerminal", "IfcAlarm"],
    "Security System":        ["IfcAlarm"],
    "Structural Frame":       ["IfcBeam", "IfcColumn", "IfcSlab"],
    "Foundation":             ["IfcFooting"],
}

# ── Space-local MEP systems ──────────────────────────────────────────────────
# Some systems are only meaningfully "present" for a space if a serving fixture
# actually bounds that space. Checking these building-wide produces false
# passes: a building with a single sanitary terminal would otherwise show
# Drainage System PASS for every kitchen and toilet (building_domain-d91). For
# the systems below, presence is decided per-space from the space's BoundedBy
# fixtures instead of `ifc.by_type` across the whole model. Distribution-only
# classes (e.g. IfcValve in a riser) are excluded — they do not give the
# bounded space the service. Systems not listed here remain building-wide.
SYSTEM_SPACE_FIXTURES: dict[str, list[str]] = {
    "Drainage System":   ["IfcSanitaryTerminal", "IfcWasteTerminal"],
    "Rough-in Plumbing": ["IfcSanitaryTerminal"],
    "Plumbing System":   ["IfcSanitaryTerminal"],
}

# ── BSOS query ────────────────────────────────────────────────────────────────

def get_requirements(db_path: Path, entity_name: str) -> list[tuple]:
    """Return (predicate, object, confidence, rationale, conditions, applicability)."""
    with sqlite3.connect(db_path) as conn:
        return conn.execute("""
            SELECT a.predicate, e2.name, a.confidence,
                   a.rationale, a.conditions, a.applicability
            FROM assertions a
            JOIN entities e  ON e.id  = a.subject_id
            JOIN entities e2 ON e2.id = a.object_id
            WHERE e.name = ?
              AND a.predicate IN ('requires', 'depends_on')
            ORDER BY a.confidence DESC
        """, (entity_name,)).fetchall()


def get_constraints(db_path: Path, entity_name: str) -> list[tuple]:
    """Return (constraint_type, rule, confidence) for an entity."""
    with sqlite3.connect(db_path) as conn:
        return conn.execute("""
            SELECT c.constraint_type, c.rule, c.confidence
            FROM constraints c
            JOIN entities e ON e.id = c.subject_id
            WHERE e.name = ?
              AND c.status != 'deprecated'
            ORDER BY c.confidence DESC
        """, (entity_name,)).fetchall()


def get_spatial_relations(db_path: Path, entity_name: str) -> list[tuple]:
    """Return (relation, object_name, confidence) for an entity."""
    with sqlite3.connect(db_path) as conn:
        return conn.execute("""
            SELECT sr.relation, e2.name, sr.confidence
            FROM spatial_relations sr
            JOIN entities e  ON e.id  = sr.subject_id
            JOIN entities e2 ON e2.id = sr.object_id
            WHERE e.name = ?
              AND sr.status != 'deprecated'
            ORDER BY sr.confidence DESC
        """, (entity_name,)).fetchall()


def get_antipatterns(db_path: Path, entity_name: str) -> list[tuple]:
    """Return (name, conditions, confidence) for an entity."""
    with sqlite3.connect(db_path) as conn:
        return conn.execute("""
            SELECT ap.name, ap.conditions, ap.confidence
            FROM antipatterns ap
            JOIN entities e ON e.id = ap.subject_id
            WHERE e.name = ?
              AND ap.status != 'deprecated'
            ORDER BY ap.confidence DESC
        """, (entity_name,)).fetchall()


def get_entity_type(db_path: Path, entity_name: str) -> str | None:
    """entity_type of an exact (already-resolved) bsos entity name, or None."""
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("""
            SELECT entity_type FROM entities WHERE name = ? AND status != 'merged'
        """, (entity_name,)).fetchone()
        return row[0] if row else None


def get_process_predecessors(db_path: Path, entity_name: str) -> list[tuple]:
    """Direct process-sequence predecessors of entity_name.

    Returns (name, hard_constraint, confidence, rationale) — the activities/
    components BSOS says must (hard_constraint) or should precede
    entity_name. Deduplicated to the single highest-confidence edge per
    predecessor, since the same predecessor->successor pair can be asserted
    from more than one source_model.
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("""
            SELECT e.name, pr.hard_constraint, pr.confidence, pr.rationale
            FROM process_relations pr
            JOIN entities e  ON e.id = pr.predecessor_id
            JOIN entities e2 ON e2.id = pr.successor_id
            WHERE e2.name = ?
              AND pr.status != 'deprecated'
            ORDER BY pr.confidence DESC
        """, (entity_name,)).fetchall()
    best: dict[str, tuple] = {}
    for name, hard, conf, rationale in rows:
        if name not in best or conf > best[name][2]:
            best[name] = (name, bool(hard), conf, rationale)
    return sorted(best.values(), key=lambda r: -r[2])


def semantic_match_entity(session: Session, query: str, entity_type: str | None = None,
                          min_score: float = SPACE_MATCH_MIN_SCORE, _embedder=None) -> str | None:
    """Best-matching bsos entity name for free text, or None below min_score.

    Reuses the same embedding-similarity ranking as the bsos `search_entities`
    MCP tool, so callers resolve arbitrary IFC names/tags against the
    knowledge graph instead of a hardcoded name lookup table.
    """
    if not query:
        return None
    result = search_entities_tool(session, query, max_results=5, min_score=min_score,
                                  _embedder=_embedder)
    for r in result["results"]:
        if entity_type is None or r["entity_type"] == entity_type:
            return r["name"]
    return None


# ── IFC helpers ───────────────────────────────────────────────────────────────

_space_entity_cache: dict[int, str | None] = {}


def resolve_space_entity(session: Session, space, _embedder=None) -> str | None:
    """Resolve an IfcSpace to its best-matching bsos 'space' entity name.

    Prefers the EPset_Topology.Usage tag (e.g. 'kitchen-1' -> 'kitchen') when
    present, falling back to the space's own LongName/Name. Cached per space id
    since each lookup is an embedding comparison.
    """
    sid = space.id()
    if sid not in _space_entity_cache:
        psets = ifcopenshell.util.element.get_psets(space)
        usage = psets.get("EPset_Topology", {}).get("Usage", "")
        query = usage.split("-")[0].strip() if usage else (space.LongName or space.Name or "")
        _space_entity_cache[sid] = semantic_match_entity(
            session, query, entity_type="space", _embedder=_embedder)
    return _space_entity_cache[sid]


def _collect_layer_materials(element, out: set[str]) -> None:
    mat = ifcopenshell.util.element.get_material(element)
    if mat is None:
        return
    if mat.is_a("IfcMaterialLayerSetUsage"):
        for layer in mat.ForLayerSet.MaterialLayers:
            if layer.Material:
                out.add(layer.Material.Name.lower())
    elif mat.is_a("IfcMaterial"):
        out.add(mat.Name.lower())


def get_floor_materials(space) -> set[str]:
    mats: set[str] = set()
    for rel in getattr(space, "ContainsElements", []):
        for elem in getattr(rel, "RelatedElements", []):
            if elem.is_a("IfcCovering"):
                _collect_layer_materials(elem, mats)
    for rel in getattr(space, "BoundedBy", []):
        elem = getattr(rel, "RelatedBuildingElement", None)
        if elem and elem.is_a("IfcCovering"):
            _collect_layer_materials(elem, mats)
    return mats


def count_bounded_by_type(space, ifc_class: str) -> int:
    n = 0
    for rel in getattr(space, "BoundedBy", []):
        elem = getattr(rel, "RelatedBuildingElement", None)
        if elem and elem.is_a(ifc_class):
            n += 1
    return n


_mep_cache: dict[str, bool] = {}

def mep_present(ifc, system: str) -> bool:
    if system not in _mep_cache:
        types = MEP_PRESENCE.get(system, [])
        _mep_cache[system] = any(ifc.by_type(t) for t in types)
    return _mep_cache[system]


def space_has_system(space, system: str) -> bool:
    """True if a fixture serving `system` bounds this space.

    Used for space-local MEP systems (SYSTEM_SPACE_FIXTURES) so that, e.g.,
    Drainage passes for a space only when a sanitary/waste terminal appears in
    its BoundedBy relations — not merely because one exists elsewhere in the
    model. Returns False for systems that are not space-local.
    """
    return any(count_bounded_by_type(space, c)
               for c in SYSTEM_SPACE_FIXTURES.get(system, ()))


def system_present_for_space(ifc, space, system: str) -> bool:
    """Per-space presence: space-local fixtures for plumbing/drainage,
    building-wide presence for distribution systems."""
    if system in SYSTEM_SPACE_FIXTURES:
        return space_has_system(space, system)
    return mep_present(ifc, system)


_GEOM_SETTINGS = ifcopenshell.geom.settings()
_geom_cache: dict[int, tuple[float | None, float | None]] = {}


def _footprint_area(geom) -> float:
    """Plan-projected floor area (m²) from a triangulated space mesh.

    Sums the XY-projected area of upward-facing triangles only. Unlike a
    bounding box this is correct for L-shaped rooms and rooms whose plan is not
    aligned to the X/Y axes — a rotation about the vertical axis leaves the
    horizontal projection of the floor face unchanged.
    """
    v, idx = geom.verts, geom.faces
    total = 0.0
    for i in range(0, len(idx), 3):
        a, b, c = idx[i], idx[i + 1], idx[i + 2]
        cross = ((v[3 * b] - v[3 * a]) * (v[3 * c + 1] - v[3 * a + 1])
                 - (v[3 * b + 1] - v[3 * a + 1]) * (v[3 * c] - v[3 * a]))
        if cross > 0:  # upward-facing → contributes to the footprint
            total += cross
    return total / 2.0


def space_dimensions(space) -> tuple[float | None, float | None]:
    """(floor_area_m2, ceiling_height_m) for a space, or (None, None).

    Height is the world-Z extent, so it too is independent of plan rotation.
    Cached per space id — shape creation is the expensive part of the report.
    """
    sid = space.id()
    if sid not in _geom_cache:
        dims: tuple[float | None, float | None] = (None, None)
        if getattr(space, "Representation", None) is not None:
            try:
                shape = ifcopenshell.geom.create_shape(_GEOM_SETTINGS, space)
                geom = shape.geometry  # keep `shape` alive while reading buffers
                dims = (_footprint_area(geom),
                        ifcopenshell.util.shape.get_z(geom))
            except Exception:
                dims = (None, None)
        _geom_cache[sid] = dims
    return _geom_cache[sid]


def wall_has_insulation(ifc) -> bool:
    for wall in ifc.by_type("IfcWall"):
        mat = ifcopenshell.util.element.get_material(wall)
        if mat and mat.is_a("IfcMaterialLayerSetUsage"):
            for layer in mat.ForLayerSet.MaterialLayers:
                if layer.Material and "insulation" in layer.Material.Name.lower():
                    return True
    return False


# ── Design-time prerequisite check (building_domain-l5w.4) ──────────────────
# IFC models don't record *when* something was built, only what's currently
# modelled — so "has this prerequisite happened yet" is approximated as "is
# there anything in the model whose name/type/material reads like it". This
# reuses the same embedding-similarity technique as resolve_space_entity, at
# the same calibrated threshold, just matched against free model text instead
# of entity embeddings looked up from the bsos entities table.
PREREQUISITE_EVIDENCE_MIN_SCORE = SPACE_MATCH_MIN_SCORE


def collect_model_evidence_texts(ifc) -> list[str]:
    """Distinct human-readable strings naming what's actually modelled.

    Drawn from every IfcProduct's Name/ObjectType and every IfcMaterial's
    Name, deduplicated case-insensitively (first-seen casing wins).
    """
    texts: dict[str, str] = {}
    for product in ifc.by_type("IfcProduct"):
        for attr in ("Name", "ObjectType"):
            val = getattr(product, attr, None)
            if val and val.strip():
                texts.setdefault(val.strip().lower(), val.strip())
    for mat in ifc.by_type("IfcMaterial"):
        if mat.Name and mat.Name.strip():
            texts.setdefault(mat.Name.strip().lower(), mat.Name.strip())
    return list(texts.values())


def _best_text_match(query: str, texts: list[str], text_vecs: "np.ndarray",
                     embedder, min_score: float = PREREQUISITE_EVIDENCE_MIN_SCORE
                     ) -> tuple[bool, str | None, float | None]:
    """Best cosine-similarity match for `query` among pre-embedded `texts`."""
    if not texts:
        return False, None, None
    q_vec = np.array(embedder.encode([query])[0], dtype=np.float32)
    q_norm = np.linalg.norm(q_vec)
    if q_norm == 0:
        return False, None, None
    best_score, best_text = -1.0, None
    for text, vec in zip(texts, text_vecs):
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        score = float(np.dot(q_vec, vec) / (q_norm * norm))
        if score > best_score:
            best_score, best_text = score, text
    return best_score >= min_score, best_text, best_score


def check_prerequisites_report(ifc_path: Path, db_path: Path, entity_name: str,
                               _embedder=None) -> dict:
    """Design-time prerequisite guardrail for authoring an IFC element.

    Call this before adding `entity_name` to a model via ifc_edit/ifc_new. It
    looks up entity_name's direct process-sequence predecessors (what BSOS
    says must/should happen first) and 'must' constraints, then checks each
    predecessor against text evidence already in the loaded model (product
    names/types, material names) so an agent can see which prerequisites are
    missing before authoring — e.g. adding interior finishes when no
    waterproofing is yet evidenced.
    """
    engine = create_db_engine(str(db_path))
    with Session(engine) as session:
        entity_row = resolve_entity(session, entity_name)
        if entity_row is None:
            return {"error": "entity_not_found", "query": entity_name}
        resolved_name = entity_row.name

    predecessors = get_process_predecessors(db_path, resolved_name)
    constraints = get_constraints(db_path, resolved_name)

    ifc = ifcopenshell.open(str(ifc_path))
    model_texts = collect_model_evidence_texts(ifc)
    embedder = _embedder or _get_embedder()
    text_vecs = (np.array(embedder.encode(model_texts), dtype=np.float32)
                if model_texts else np.empty((0, 0), dtype=np.float32))

    prereq_results = []
    for name, hard_constraint, confidence, rationale in predecessors:
        evidenced, matched_text, score = _best_text_match(name, model_texts, text_vecs, embedder)
        prereq_results.append({
            "prerequisite": name,
            "hard_constraint": hard_constraint,
            "confidence": confidence,
            "rationale": rationale,
            "evidenced": evidenced,
            "matched_text": matched_text,
            "match_score": round(score, 4) if score is not None else None,
        })

    missing = [r for r in prereq_results if not r["evidenced"]]
    missing_hard = [r for r in missing if r["hard_constraint"]]

    if missing_hard:
        recommendation = (
            f"Hold: {len(missing_hard)} hard-constraint prerequisite(s) not yet "
            "evidenced in the model"
        )
    elif missing:
        recommendation = (
            f"Proceed with caution: {len(missing)} soft prerequisite(s) not yet "
            "evidenced in the model"
        )
    else:
        recommendation = "OK to proceed"

    return {
        "entity": resolved_name,
        "prerequisites": prereq_results,
        "constraints": [
            {"constraint_type": ctype, "rule": rule, "confidence": conf}
            for ctype, rule, conf in constraints
        ],
        "summary": {
            "total_prerequisites": len(prereq_results),
            "evidenced": len(prereq_results) - len(missing),
            "missing": len(missing),
            "missing_hard_constraints": len(missing_hard),
        },
        "recommendation": recommendation,
    }


# ── Fact extraction ───────────────────────────────────────────────────────────
# Turn an IfcSpace into the model-fact dict the shared validation engine
# consumes. An MCP agent populates the same shape from the `ifc` server tools.

def build_facts(ifc, space) -> dict:
    area, height = space_dimensions(space)
    return {
        "floor_materials": sorted(get_floor_materials(space)),
        "window_count":    count_bounded_by_type(space, "IfcWindow"),
        "door_count":      count_bounded_by_type(space, "IfcDoor"),
        "systems_present": [s for s in MEP_PRESENCE
                            if system_present_for_space(ifc, space, s)],
        "has_insulation":  wall_has_insulation(ifc),
        "floor_area_m2":   area,
        "ceiling_height_m": height,
    }


def check(obj: str, facts: dict) -> tuple[str, str]:
    """Evaluate one check object via the shared engine (upper-case status)."""
    status, detail = validation.evaluate(obj, facts)
    return status.upper(), detail


# Relations that assert two spaces share a boundary / connection.
ADJACENCY_RELATIONS = {"adjacent_to", "connects_to", "accessible_from",
                       "connected_to"}


def build_adjacency(ifc) -> dict[int, set[int]]:
    """Map each IfcSpace id → set of space ids sharing a bounding element."""
    elem_to_spaces: dict[int, set[int]] = defaultdict(set)
    for sp in ifc.by_type("IfcSpace"):
        for rel in getattr(sp, "BoundedBy", []):
            el = getattr(rel, "RelatedBuildingElement", None)
            if el is not None:
                elem_to_spaces[el.id()].add(sp.id())
    adj: dict[int, set[int]] = defaultdict(set)
    for sid_set in elem_to_spaces.values():
        for a in sid_set:
            adj[a] |= (sid_set - {a})
    return adj


# ── Report ────────────────────────────────────────────────────────────────────

SYM = {"PASS": "✓", "FAIL": "✗", "UNCHECKED": "?"}
W   = 72


def _fmt_applicability(appl_json: str) -> str:
    try:
        items = json.loads(appl_json or "[]")
        return ", ".join(items) if items else ""
    except Exception:
        return ""


def _emit(label: str, confidence: float,
          space_statuses: list[tuple[str, str, str]],
          totals: dict, all_rows: list[dict], row_base: dict,
          extra: str = "", quiet: bool = False) -> None:
    """Record one check, printing it collapsed/expanded unless quiet=True.

    quiet=True is used by the `check_model` MCP tool: printing to stdout would
    corrupt the MCP server's stdio JSON-RPC stream, so tool-driven runs only
    build `all_rows` and never touch stdout.
    """
    statuses = [s for _, s, _ in space_statuses]
    if len(set(statuses)) == 1:
        status = statuses[0]
        detail = space_statuses[0][2]
        totals[status] += 1
        if not quiet:
            print(f"   {SYM[status]} [{confidence:.0%}] {label}{extra}")
            print(f"         {detail}")
        all_rows.append({**row_base, "status": status, "detail": detail})
    else:
        for sname, status, detail in space_statuses:
            totals[status] += 1
            if not quiet:
                print(f"   {SYM[status]} [{confidence:.0%}] {label}  ({sname})")
                print(f"         {detail}")
            all_rows.append({**row_base, "status": status, "detail": detail,
                             "space": sname})


def report_requirements(ifc, entity, spaces, reqs, totals, all_rows, quiet=False) -> None:
    if not reqs:
        return
    if not quiet:
        print("   REQUIREMENTS")
    req_results: dict[tuple, list[tuple[str, str, str]]] = defaultdict(list)
    for space in spaces:
        facts = build_facts(ifc, space)
        for predicate, obj, confidence, rationale, conds, appl in reqs:
            status, detail = check(obj, facts)
            req_results[(predicate, obj, confidence, appl)].append(
                (space.Name or "?", status, detail))

    for (predicate, obj, confidence, appl), space_statuses in req_results.items():
        appl_str = _fmt_applicability(appl)
        _emit(f"{predicate} {obj}", confidence, space_statuses, totals, all_rows,
              {"category": "requirement", "space_type": entity, "entity": entity,
               "predicate": predicate, "object": obj, "applicability": appl_str},
              extra=f"  [{appl_str}]" if appl_str else "", quiet=quiet)


def report_constraints(ifc, entity, spaces, constraints, totals, all_rows, quiet=False) -> None:
    if not constraints:
        return
    if not quiet:
        print("   CONSTRAINTS")
    unchecked = 0
    for ctype, rule, confidence in constraints:
        obj = validation.classify_constraint(rule, ctype)
        dim = validation.classify_dimensional_constraint(rule, ctype) if obj is None else None
        if obj is None and dim is None:
            unchecked += 1
            continue
        space_statuses = []
        for space in spaces:
            facts = build_facts(ifc, space)
            if obj is not None:
                status, detail = check(obj, facts)
            else:
                kind, threshold = dim
                status, detail = validation.evaluate_dimensional(kind, threshold, facts)
                status = status.upper()
            space_statuses.append((space.Name or "?", status, detail))
        _emit(f"{ctype} — {rule}", confidence, space_statuses, totals, all_rows,
              {"category": "constraint", "space_type": entity, "entity": entity,
               "object": obj if obj is not None else dim[0], "rule": rule}, quiet=quiet)
    if unchecked and not quiet:
        print(f"   ·  {unchecked} constraint(s) not mechanically checkable "
              f"(non-dimensional / code)")


def report_spatial(db_path, entity, spaces, rels, adj, id_to_entity,
                   entities_present, totals, all_rows, quiet=False) -> None:
    """Check adjacent_to/connects_to/accessible_from relations.

    `rels` objects are already-resolved bsos entity names (from the join in
    get_spatial_relations), so a checkable relation is simply one whose object
    is itself a bsos 'space' entity — no separate name-matching vocabulary is
    needed. PASS/FAIL is decided against the spaces actually resolved in this
    model; UNCHECKED means the target space entity exists in bsos but nothing
    in this model resolved to it.
    """
    checkable = [(rel, obj, conf) for rel, obj, conf in rels
                 if rel in ADJACENCY_RELATIONS and get_entity_type(db_path, obj) == "space"]
    if not checkable:
        return
    if not quiet:
        print("   SPATIAL")
    for rel, obj, conf in checkable:
        space_statuses = []
        for space in spaces:
            if obj not in entities_present:
                status, detail = "UNCHECKED", (
                    f"no space modelled resolves to bsos entity '{obj}'")
            else:
                neighbours = {id_to_entity.get(nid) for nid in adj.get(space.id(), set())}
                if obj in neighbours:
                    status, detail = "PASS", f"adjacent to a '{obj}' space"
                else:
                    status, detail = "FAIL", f"not adjacent to any '{obj}' space"
            space_statuses.append((space.Name or "?", status, detail))
        _emit(f"{rel} {obj}", conf, space_statuses, totals, all_rows,
              {"category": "spatial", "space_type": entity, "entity": entity,
               "relation": rel, "object": obj}, quiet=quiet)
    skipped = len(rels) - len(checkable)
    if skipped and not quiet:
        print(f"   ·  {skipped} relation(s) to non-spatial / unmodelled "
              f"objects skipped")


def report_antipatterns(ifc, entity, spaces, aps, totals, all_rows, quiet=False) -> None:
    if not aps:
        return
    if not quiet:
        print("   ANTIPATTERNS")
    flagged = 0
    for space in spaces:
        signals = validation.antipattern_signals(build_facts(ifc, space))
        for name, conds, conf in aps:
            topic = validation.antipattern_triggered(name, conds, signals)
            if topic:
                flagged += 1
                totals["FAIL"] += 1
                if not quiet:
                    print(f"   {SYM['FAIL']} [{conf:.0%}] {name}  ({space.Name or '?'})")
                    print(f"         model exhibits '{topic}' failure signal")
                all_rows.append({
                    "category": "antipattern", "space_type": entity,
                    "entity": entity, "object": name,
                    "status": "FAIL",
                    "detail": f"'{topic}' failure signal present",
                    "space": space.Name or "?"})
    if not flagged:
        totals["PASS"] += 1
        if not quiet:
            print(f"   {SYM['PASS']} none of {len(aps)} known failure "
                  f"condition(s) detected")
        all_rows.append({
            "category": "antipattern", "space_type": entity, "entity": entity,
            "object": "(none triggered)", "status": "PASS",
            "detail": f"{len(aps)} antipattern(s) checked, none triggered"})


def summarize(all_rows: list[dict]) -> dict:
    """Aggregate {totals, by_category, failures} from a run_report() result.

    Shared by the CLI's printed summary and the `check_model` MCP tool, so the
    two presentations of a report can never disagree.
    """
    totals = {"PASS": 0, "FAIL": 0, "UNCHECKED": 0}
    for r in all_rows:
        totals[r["status"]] += 1

    by_category = {}
    for cat, label in (("requirement", "requirements"), ("constraint", "constraints"),
                       ("spatial", "spatial"), ("antipattern", "antipatterns")):
        rows = [r for r in all_rows if r.get("category") == cat]
        if rows:
            by_category[label] = {
                "pass": sum(r["status"] == "PASS" for r in rows),
                "fail": sum(r["status"] == "FAIL" for r in rows),
                "unchecked": sum(r["status"] == "UNCHECKED" for r in rows),
            }

    seen: set[tuple] = set()
    failures = []
    for r in all_rows:
        if r["status"] != "FAIL":
            continue
        key = (r["entity"], r.get("object", ""), r["detail"])
        if key in seen:
            continue
        seen.add(key)
        failures.append({"entity": r["entity"], "object": r.get("object", ""),
                         "detail": r["detail"]})

    return {
        "total": sum(totals.values()),
        "pass": totals["PASS"],
        "fail": totals["FAIL"],
        "unchecked": totals["UNCHECKED"],
        "by_category": by_category,
        "failures": failures,
    }


def run_report(ifc_path: Path = IFC_PATH, db_path: Path = BSOS_DB, _embedder=None,
              quiet: bool = False) -> list[dict]:
    """Run the full compliance report. Returns the flat list of check rows.

    quiet=True suppresses all stdout output (used by the `check_model` MCP
    tool — printing would corrupt the MCP server's stdio JSON-RPC stream) and
    skips geometry/facts work for spaces with no bsos knowledge to check.
    """
    ifc = ifcopenshell.open(str(ifc_path))
    _mep_cache.clear()
    _geom_cache.clear()
    _space_entity_cache.clear()

    engine = create_db_engine(str(db_path))
    spaces_by_entity: dict[str, list] = defaultdict(list)
    with Session(engine) as session:
        for space in ifc.by_type("IfcSpace"):
            entity_name = resolve_space_entity(session, space, _embedder=_embedder)
            if entity_name:
                spaces_by_entity[entity_name].append(space)

    adj = build_adjacency(ifc)
    id_to_entity = {sp.id(): e for e, sps in spaces_by_entity.items() for sp in sps}
    entities_present = set(spaces_by_entity)

    totals   = {"PASS": 0, "FAIL": 0, "UNCHECKED": 0}
    all_rows: list[dict] = []

    if not quiet:
        print(f"\n{'='*W}")
        print(f"  IFC × BSOS  Compliance Report")
        print(f"  Model : {Path(ifc_path).name}")
        print(f"{'='*W}\n")

    for entity in sorted(spaces_by_entity):
        reqs   = get_requirements(db_path, entity)
        cons   = get_constraints(db_path, entity)
        rels   = get_spatial_relations(db_path, entity)
        aps    = get_antipatterns(db_path, entity)
        spaces = spaces_by_entity[entity]
        if not (reqs or cons or rels or aps):
            continue

        if not quiet:
            space_names = ", ".join(s.Name or "?" for s in sorted(spaces, key=lambda s: s.Name or ""))
            print(f"▶  {entity}  ({len(spaces)} space(s): {space_names})")
            print(f"   {'─'*(W-3)}")

        report_requirements(ifc, entity, spaces, reqs, totals, all_rows, quiet=quiet)
        report_constraints(ifc, entity, spaces, cons, totals, all_rows, quiet=quiet)
        report_spatial(db_path, entity, spaces, rels, adj, id_to_entity,
                       entities_present, totals, all_rows, quiet=quiet)
        report_antipatterns(ifc, entity, spaces, aps, totals, all_rows, quiet=quiet)
        if not quiet:
            print()

    if not quiet:
        summary = summarize(all_rows)
        print(f"{'='*W}")
        print(
            f"  Checks: {summary['total']}   "
            f"✓ PASS {summary['pass']}   "
            f"✗ FAIL {summary['fail']}   "
            f"? UNCHECKED {summary['unchecked']}"
        )
        for label, counts in summary["by_category"].items():
            print(f"    {label:<13} ✓ {counts['pass']}   ✗ {counts['fail']}   ? {counts['unchecked']}")
        print(f"{'='*W}\n")

        if summary["failures"]:
            print("FAILURES")
            print(f"{'─'*W}")
            for f in summary["failures"]:
                print(f"  ✗  {f['entity']} — {f['object']}")
                print(f"     {f['detail']}")
            print()

    return all_rows


if __name__ == "__main__":
    ifc_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else IFC_PATH
    db_arg  = Path(sys.argv[2]) if len(sys.argv) > 2 else BSOS_DB
    run_report(ifc_arg, db_arg)
