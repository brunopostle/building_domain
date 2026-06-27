#!/usr/bin/env python3
"""IFC × BSOS compliance report.

For every space type in an IFC model this checks four BSOS knowledge layers
against what is actually modelled:

  * REQUIREMENTS  — requires/depends_on assertions (materials, elements, MEP)
  * CONSTRAINTS   — `must` rules from the constraints table
  * SPATIAL       — adjacent_to/connects_to/accessible_from relations vs the
                    IFC spatial structure (shared bounding walls)
  * ANTIPATTERNS  — known failure conditions, flagged only when the model
                    affirmatively exhibits the negative signal

Usage:
    python scripts/ifc_compliance_report.py [path/to/model.ifc] [path/to/bsos.db]
"""
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import ifcopenshell
import ifcopenshell.util.element

# The deterministic check engine is shared with the bsos `validate_element` MCP
# tool, so the report and the tool can never disagree. This script's job is to
# extract `facts` from the IFC model and hand them to that engine.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bsos import validation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IFC_PATH = ROOT / "_test.ifc"
BSOS_DB  = ROOT / "bsos.db"

# ── Space type → BSOS entity ─────────────────────────────────────────────────
SPACE_TO_ENTITY = {
    "kitchen":     "Kitchen",
    "living":      "Living Room",
    "circulation": "Hallway / Circulation Corridor",
    "toilet":      "Toilet / WC",
    "stair":       "Staircase / Stair Hall",
    "retail":      "Retail Unit / Shop Front",
}

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


# ── IFC helpers ───────────────────────────────────────────────────────────────

def get_space_usage(space) -> str | None:
    psets = ifcopenshell.util.element.get_psets(space)
    usage = psets.get("EPset_Topology", {}).get("Usage", "")
    if usage:
        return usage.lower().split("-")[0].strip()
    name = (space.Name or "").lower()
    for key in SPACE_TO_ENTITY:
        if key in name:
            return key
    return None


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


def wall_has_insulation(ifc) -> bool:
    for wall in ifc.by_type("IfcWall"):
        mat = ifcopenshell.util.element.get_material(wall)
        if mat and mat.is_a("IfcMaterialLayerSetUsage"):
            for layer in mat.ForLayerSet.MaterialLayers:
                if layer.Material and "insulation" in layer.Material.Name.lower():
                    return True
    return False


# ── Fact extraction ───────────────────────────────────────────────────────────
# Turn an IfcSpace into the model-fact dict the shared validation engine
# consumes. An MCP agent populates the same shape from the `ifc` server tools.

def build_facts(ifc, space) -> dict:
    return {
        "floor_materials": sorted(get_floor_materials(space)),
        "window_count":    count_bounded_by_type(space, "IfcWindow"),
        "door_count":      count_bounded_by_type(space, "IfcDoor"),
        "systems_present": [s for s in MEP_PRESENCE
                            if system_present_for_space(ifc, space, s)],
        "has_insulation":  wall_has_insulation(ifc),
    }


def check(obj: str, facts: dict) -> tuple[str, str]:
    """Evaluate one check object via the shared engine (upper-case status)."""
    status, detail = validation.evaluate(obj, facts)
    return status.upper(), detail


# ── Spatial relations: BSOS object name → modelled space usage ───────────────
# Only relations to objects that are themselves modelled space types can be
# checked; relations to furniture/structure (Countertop, Wall, Basement) are
# skipped.
SPATIAL_OBJECT_TO_USAGE: dict[str, str] = {
    "kitchen": "kitchen",
    "living": "living",
    "hallway": "circulation",
    "corridor": "circulation",
    "circulation": "circulation",
    "staircase": "stair",
    "stair hall": "stair",
    "stair": "stair",
    "toilet": "toilet",
    "wc": "toilet",
    "retail": "retail",
    "shop": "retail",
}

# Relations that assert two spaces share a boundary / connection.
ADJACENCY_RELATIONS = {"adjacent_to", "connects_to", "accessible_from",
                       "connected_to"}


def object_to_usage(object_name: str) -> str | None:
    n = object_name.lower()
    for key, usage in SPATIAL_OBJECT_TO_USAGE.items():
        if key in n:
            return usage
    return None


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
          extra: str = "") -> None:
    """Print one check, collapsed if every space agrees else expanded."""
    statuses = [s for _, s, _ in space_statuses]
    if len(set(statuses)) == 1:
        status = statuses[0]
        detail = space_statuses[0][2]
        totals[status] += 1
        print(f"   {SYM[status]} [{confidence:.0%}] {label}{extra}")
        print(f"         {detail}")
        all_rows.append({**row_base, "status": status, "detail": detail})
    else:
        for sname, status, detail in space_statuses:
            totals[status] += 1
            print(f"   {SYM[status]} [{confidence:.0%}] {label}  ({sname})")
            print(f"         {detail}")
            all_rows.append({**row_base, "status": status, "detail": detail,
                             "space": sname})


def report_requirements(ifc, usage, entity, spaces, reqs,
                        totals, all_rows) -> None:
    if not reqs:
        return
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
              {"category": "requirement", "space_type": usage, "entity": entity,
               "predicate": predicate, "object": obj, "applicability": appl_str},
              extra=f"  [{appl_str}]" if appl_str else "")


def report_constraints(ifc, usage, entity, spaces, constraints,
                       totals, all_rows) -> None:
    if not constraints:
        return
    print("   CONSTRAINTS")
    unchecked = 0
    for ctype, rule, confidence in constraints:
        obj = validation.classify_constraint(rule, ctype)
        if obj is None:
            unchecked += 1
            continue
        space_statuses = []
        for space in spaces:
            status, detail = check(obj, build_facts(ifc, space))
            space_statuses.append((space.Name or "?", status, detail))
        _emit(f"{ctype} — {rule}", confidence, space_statuses, totals, all_rows,
              {"category": "constraint", "space_type": usage, "entity": entity,
               "object": obj, "rule": rule})
    if unchecked:
        print(f"   ·  {unchecked} constraint(s) not mechanically checkable "
              f"(dimensional / code)")


def report_spatial(ifc, usage, entity, spaces, rels, adj, id_to_usage,
                   usages_present, totals, all_rows) -> None:
    checkable = [(rel, obj, conf) for rel, obj, conf in rels
                 if rel in ADJACENCY_RELATIONS and object_to_usage(obj)]
    if not checkable:
        return
    print("   SPATIAL")
    for rel, obj, conf in checkable:
        target = object_to_usage(obj)
        space_statuses = []
        for space in spaces:
            if target not in usages_present:
                status, detail = "UNCHECKED", (
                    f"no '{target}' space modelled to verify against")
            else:
                neighbours = {id_to_usage.get(nid) for nid in adj.get(space.id(), set())}
                if target in neighbours:
                    status, detail = "PASS", f"adjacent to a '{target}' space"
                else:
                    status, detail = "FAIL", f"not adjacent to any '{target}' space"
            space_statuses.append((space.Name or "?", status, detail))
        _emit(f"{rel} {obj}", conf, space_statuses, totals, all_rows,
              {"category": "spatial", "space_type": usage, "entity": entity,
               "relation": rel, "object": obj})
    skipped = len(rels) - len(checkable)
    if skipped:
        print(f"   ·  {skipped} relation(s) to non-spatial / unmodelled "
              f"objects skipped")


def report_antipatterns(ifc, usage, entity, spaces, aps,
                        totals, all_rows) -> None:
    if not aps:
        return
    print("   ANTIPATTERNS")
    flagged = 0
    for space in spaces:
        signals = validation.antipattern_signals(build_facts(ifc, space))
        for name, conds, conf in aps:
            topic = validation.antipattern_triggered(name, conds, signals)
            if topic:
                flagged += 1
                totals["FAIL"] += 1
                print(f"   {SYM['FAIL']} [{conf:.0%}] {name}  ({space.Name or '?'})")
                print(f"         model exhibits '{topic}' failure signal")
                all_rows.append({
                    "category": "antipattern", "space_type": usage,
                    "entity": entity, "object": name,
                    "status": "FAIL",
                    "detail": f"'{topic}' failure signal present",
                    "space": space.Name or "?"})
    if not flagged:
        totals["PASS"] += 1
        print(f"   {SYM['PASS']} none of {len(aps)} known failure "
              f"condition(s) detected")
        all_rows.append({
            "category": "antipattern", "space_type": usage, "entity": entity,
            "object": "(none triggered)", "status": "PASS",
            "detail": f"{len(aps)} antipattern(s) checked, none triggered"})


def run_report(ifc_path: Path = IFC_PATH, db_path: Path = BSOS_DB) -> list[dict]:
    ifc = ifcopenshell.open(str(ifc_path))
    _mep_cache.clear()

    spaces_by_usage: dict[str, list] = defaultdict(list)
    for space in ifc.by_type("IfcSpace"):
        usage = get_space_usage(space)
        if usage and usage in SPACE_TO_ENTITY:
            spaces_by_usage[usage].append(space)

    adj = build_adjacency(ifc)
    id_to_usage = {sp.id(): u for u, sps in spaces_by_usage.items() for sp in sps}
    usages_present = set(spaces_by_usage)

    totals   = {"PASS": 0, "FAIL": 0, "UNCHECKED": 0}
    all_rows: list[dict] = []

    print(f"\n{'='*W}")
    print(f"  IFC × BSOS  Compliance Report")
    print(f"  Model : {Path(ifc_path).name}")
    print(f"{'='*W}\n")

    for usage in sorted(spaces_by_usage):
        entity = SPACE_TO_ENTITY[usage]
        reqs   = get_requirements(db_path, entity)
        cons   = get_constraints(db_path, entity)
        rels   = get_spatial_relations(db_path, entity)
        aps    = get_antipatterns(db_path, entity)
        spaces = spaces_by_usage[usage]
        if not (reqs or cons or rels or aps):
            continue

        space_names = ", ".join(s.Name or "?" for s in sorted(spaces, key=lambda s: s.Name or ""))
        print(f"▶  {entity}  ({len(spaces)} space(s): {space_names})")
        print(f"   {'─'*(W-3)}")

        report_requirements(ifc, usage, entity, spaces, reqs, totals, all_rows)
        report_constraints(ifc, usage, entity, spaces, cons, totals, all_rows)
        report_spatial(ifc, usage, entity, spaces, rels, adj, id_to_usage,
                       usages_present, totals, all_rows)
        report_antipatterns(ifc, usage, entity, spaces, aps, totals, all_rows)
        print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"{'='*W}")
    total = sum(totals.values())
    print(
        f"  Checks: {total}   "
        f"✓ PASS {totals['PASS']}   "
        f"✗ FAIL {totals['FAIL']}   "
        f"? UNCHECKED {totals['UNCHECKED']}"
    )
    # Per-layer breakdown
    for cat, label in (("requirement", "requirements"), ("constraint", "constraints"),
                       ("spatial", "spatial"), ("antipattern", "antipatterns")):
        rows = [r for r in all_rows if r.get("category") == cat]
        if rows:
            p = sum(r["status"] == "PASS" for r in rows)
            f = sum(r["status"] == "FAIL" for r in rows)
            u = sum(r["status"] == "UNCHECKED" for r in rows)
            print(f"    {label:<13} ✓ {p}   ✗ {f}   ? {u}")
    print(f"{'='*W}\n")

    failures = [r for r in all_rows if r["status"] == "FAIL"]
    if failures:
        print("FAILURES")
        print(f"{'─'*W}")
        seen: set[tuple] = set()
        for r in failures:
            key = (r["entity"], r["object"], r["detail"])
            if key in seen:
                continue
            seen.add(key)
            print(f"  ✗  {r['entity']} — {r.get('object', '')}")
            print(f"     {r['detail']}")
        print()

    return all_rows


if __name__ == "__main__":
    ifc_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else IFC_PATH
    db_arg  = Path(sys.argv[2]) if len(sys.argv) > 2 else BSOS_DB
    run_report(ifc_arg, db_arg)
