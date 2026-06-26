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

# ── Material quality sets ────────────────────────────────────────────────────
SLIP_RESISTANT = {"tiles", "concrete", "stone", "vinyl", "screed",
                  "terrazzo", "ceramic", "epoxy", "resin", "linoleum"}
WATERPROOF     = {"tiles", "concrete", "stone", "vinyl", "ceramic",
                  "epoxy", "resin", "membrane", "linoleum"}

# ── IFC element types that represent each system ─────────────────────────────
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

# Objects whose check is flooring-material based
FLOORING_QUALITY: dict[str, set[str]] = {
    "Anti-slip Flooring": SLIP_RESISTANT,
    "Waterproof Flooring": WATERPROOF,
}

# Objects that cannot be verified from IFC geometry/materials alone
UNCHECKED_OBJECTS = {
    "Countertop Material", "Refrigeration Unit", "Nosing Profile",
    "Door Hardware", "Stringer", "Landing", "Ceiling",
    "Countertop", "Cabinetry",
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


def wall_has_insulation(ifc) -> bool:
    for wall in ifc.by_type("IfcWall"):
        mat = ifcopenshell.util.element.get_material(wall)
        if mat and mat.is_a("IfcMaterialLayerSetUsage"):
            for layer in mat.ForLayerSet.MaterialLayers:
                if layer.Material and "insulation" in layer.Material.Name.lower():
                    return True
    return False


# ── Per-requirement check ─────────────────────────────────────────────────────

def check(ifc, space, req_object: str, floor_mats: set[str]) -> tuple[str, str]:
    """Return (status, detail)  where status ∈ {PASS, FAIL, UNCHECKED}."""
    obj = req_object

    if obj == "Flooring":
        if floor_mats:
            return "PASS", f"floor material present: {', '.join(sorted(floor_mats))}"
        return "FAIL", "no floor covering found"

    if obj in FLOORING_QUALITY:
        approved = FLOORING_QUALITY[obj]
        good = floor_mats & approved
        if good:
            return "PASS", f"approved material: {', '.join(sorted(good))}"
        if floor_mats:
            return "FAIL", (
                f"floor is {', '.join(sorted(floor_mats))} "
                f"— not in approved set {{{', '.join(sorted(approved))}}}"
            )
        return "FAIL", "no floor covering found"

    if obj == "Windows":
        n = count_bounded_by_type(space, "IfcWindow")
        return ("PASS", f"{n} window(s)") if n else ("FAIL", "no windows found")

    if obj in ("Doors", "Entrance Door", "External Door"):
        n = count_bounded_by_type(space, "IfcDoor")
        return ("PASS", f"{n} door(s)") if n else ("FAIL", "no doors found")

    if obj == "Ventilation System or Operable Window":
        win = count_bounded_by_type(space, "IfcWindow")
        if mep_present(ifc, "Ventilation System") or win:
            return "PASS", (
                "ventilation system present"
                if mep_present(ifc, "Ventilation System")
                else f"{win} operable window(s) in space"
            )
        return "FAIL", "no ventilation system and no window in space"

    if obj in MEP_PRESENCE:
        present = mep_present(ifc, obj)
        return (
            ("PASS", "system elements present in model")
            if present
            else ("FAIL", "no MEP elements of this type found in model")
        )

    if obj == "Insulation":
        has = wall_has_insulation(ifc)
        return (
            ("PASS", "insulation layer found in wall assemblies")
            if has
            else ("FAIL", "no insulation layer in any wall assembly")
        )

    if obj in UNCHECKED_OBJECTS:
        return "UNCHECKED", "requires non-geometric IFC data"

    return "UNCHECKED", f"no check implemented for '{obj}'"


# ── Constraints: free-text rule → checkable object ───────────────────────────
#
# Constraint rules are natural-language `must` / `must_not` statements. Only the
# subset that maps to a physical signal modellable in IFC is mechanically
# checkable; the rest (clear widths, ceiling heights, riser ratios, fire
# ratings, electrical circuits) are dimensional/code rules left UNCHECKED.

def classify_constraint(rule: str, ctype: str) -> str | None:
    """Map a constraint rule to a `check()` object, or None if not checkable."""
    if ctype == "must_not":
        # Prohibitions here are geometric/code (clearances, projections, sill
        # heights) — none reduce to a presence test against this model.
        return None
    r = rule.lower()
    if "floor" in r and ("waterproof" in r or "water-resistant" in r
                         or "water resistant" in r):
        return "Waterproof Flooring"
    if "slip" in r and ("resist" in r or "anti-slip" in r or "non-slip" in r):
        return "Anti-slip Flooring"
    if "drainage" in r or "wastewater" in r or "waste water" in r:
        return "Drainage System"
    if "water supply" in r or "plumbing supply" in r or (
        "plumbing" in r and "connection" in r):
        return "Plumbing System"
    if "ventilation" in r or "exhaust" in r or "air change" in r:
        return "Ventilation System or Operable Window"
    if "window" in r or "glazing" in r or "daylight" in r:
        return "Windows"
    if "entrance" in r or "door" in r or "egress" in r or "exit" in r:
        return "Doors"
    if "insulation" in r:
        return "Insulation"
    return None


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


# ── Antipatterns: negative-signal detection ──────────────────────────────────
# Antipattern conditions are fine-grained free text. We flag one only when its
# topic keyword co-occurs with a negative signal we can confirm from the model,
# so a complete model yields no false positives.
ANTIPATTERN_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ventilation": ("unvented", "poorly vented", "no ventilation", "stale air",
                    "stuffy", "no exhaust", "no airflow"),
    "drainage":    ("no drainage", "no drain", "missing drainage", "no plumbing",
                    "standing water", "sewage backup"),
    "window":      ("no window", "windowless", "no daylight", "no natural light",
                    "no view", "no exterior wall"),
    "flooring":    ("no floor finish", "bare subfloor", "missing floor"),
}


def antipattern_signals(ifc, space, floor_mats: set[str]) -> dict[str, bool]:
    """Negative signals that hold for this space (True = failure condition)."""
    win = count_bounded_by_type(space, "IfcWindow")
    return {
        "ventilation": not (mep_present(ifc, "Ventilation System") or win),
        "drainage":    not mep_present(ifc, "Drainage System"),
        "window":      win == 0,
        "flooring":    not floor_mats,
    }


def antipattern_triggered(name: str, conditions: str | None,
                          signals: dict[str, bool]) -> str | None:
    """Return the triggering topic if a confirmable failure signal matches."""
    text = (name + " " + (conditions or "")).lower()
    for topic, keywords in ANTIPATTERN_TOPIC_KEYWORDS.items():
        if signals.get(topic) and any(kw in text for kw in keywords):
            return topic
    return None


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
        floor_mats = get_floor_materials(space)
        for predicate, obj, confidence, rationale, conds, appl in reqs:
            status, detail = check(ifc, space, obj, floor_mats)
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
        obj = classify_constraint(rule, ctype)
        if obj is None:
            unchecked += 1
            continue
        space_statuses = []
        for space in spaces:
            floor_mats = get_floor_materials(space)
            status, detail = check(ifc, space, obj, floor_mats)
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
        floor_mats = get_floor_materials(space)
        signals = antipattern_signals(ifc, space, floor_mats)
        for name, conds, conf in aps:
            topic = antipattern_triggered(name, conds, signals)
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
