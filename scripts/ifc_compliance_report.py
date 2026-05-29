#!/usr/bin/env python3
"""IFC × BSOS requirements compliance report.

For every space type in an IFC model, fetches BSOS requires/depends_on
assertions and checks them against what is actually modelled.

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
    "Drainage System":        ["IfcPipeSegment", "IfcSanitaryTerminal",
                               "IfcWasteTerminal", "IfcPipeFitting"],
    "Rough-in Plumbing":      ["IfcPipeSegment", "IfcValve", "IfcSanitaryTerminal"],
    "Plumbing System":        ["IfcPipeSegment", "IfcValve", "IfcSanitaryTerminal"],
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


# ── Report ────────────────────────────────────────────────────────────────────

SYM = {"PASS": "✓", "FAIL": "✗", "UNCHECKED": "?"}
W   = 72


def _fmt_applicability(appl_json: str) -> str:
    try:
        items = json.loads(appl_json or "[]")
        return ", ".join(items) if items else ""
    except Exception:
        return ""


def run_report(ifc_path: Path = IFC_PATH, db_path: Path = BSOS_DB) -> list[dict]:
    ifc = ifcopenshell.open(str(ifc_path))
    _mep_cache.clear()

    spaces_by_usage: dict[str, list] = defaultdict(list)
    for space in ifc.by_type("IfcSpace"):
        usage = get_space_usage(space)
        if usage and usage in SPACE_TO_ENTITY:
            spaces_by_usage[usage].append(space)

    totals   = {"PASS": 0, "FAIL": 0, "UNCHECKED": 0}
    all_rows: list[dict] = []

    print(f"\n{'='*W}")
    print(f"  IFC × BSOS  Compliance Report — Requirements")
    print(f"  Model : {Path(ifc_path).name}")
    print(f"{'='*W}\n")

    for usage in sorted(spaces_by_usage):
        entity = SPACE_TO_ENTITY[usage]
        reqs   = get_requirements(db_path, entity)
        spaces = spaces_by_usage[usage]
        if not reqs:
            continue

        space_names = ", ".join(s.Name or "?" for s in sorted(spaces, key=lambda s: s.Name or ""))
        print(f"▶  {entity}  ({len(spaces)} space(s): {space_names})")
        print(f"   {'─'*(W-3)}")

        # Aggregate results per (req_object) across all spaces of this type.
        # If every space gives the same status, show once; otherwise expand.
        req_results: dict[tuple, list[tuple[str, str, str]]] = defaultdict(list)
        for space in spaces:
            floor_mats = get_floor_materials(space)
            for predicate, obj, confidence, rationale, conds, appl in reqs:
                status, detail = check(ifc, space, obj, floor_mats)
                req_results[(predicate, obj, confidence, appl)].append(
                    (space.Name or "?", status, detail)
                )

        for (predicate, obj, confidence, appl), space_statuses in req_results.items():
            statuses = [s for _, s, _ in space_statuses]
            details  = [d for _, _, d in space_statuses]
            appl_str = _fmt_applicability(appl)

            if len(set(statuses)) == 1:
                # All spaces agree — show once
                status = statuses[0]
                detail = details[0]
                sym    = SYM[status]
                totals[status] += 1
                appl_note = f"  [{appl_str}]" if appl_str else ""
                print(f"   {sym} [{confidence:.0%}] {predicate} {obj}{appl_note}")
                print(f"         {detail}")
                all_rows.append({
                    "space_type": usage, "entity": entity,
                    "predicate": predicate, "object": obj,
                    "status": status, "detail": detail,
                    "applicability": appl_str,
                })
            else:
                # Spaces differ — expand per space
                for space_name, status, detail in space_statuses:
                    sym = SYM[status]
                    totals[status] += 1
                    print(f"   {sym} [{confidence:.0%}] {predicate} {obj}  ({space_name})")
                    print(f"         {detail}")
                    all_rows.append({
                        "space_type": usage, "entity": entity,
                        "predicate": predicate, "object": obj,
                        "status": status, "detail": detail,
                        "space": space_name, "applicability": appl_str,
                    })
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
    print(f"{'='*W}\n")

    failures = [r for r in all_rows if r["status"] == "FAIL"]
    if failures:
        print("FAILURES")
        print(f"{'─'*W}")
        seen: set[tuple] = set()
        for r in failures:
            key = (r["object"], r["detail"])
            if key in seen:
                continue
            seen.add(key)
            appl = f"  [{r['applicability']}]" if r.get("applicability") else ""
            print(f"  ✗  {r['entity']} — {r['predicate']} {r['object']}{appl}")
            print(f"     {r['detail']}")
        print()

    return all_rows


if __name__ == "__main__":
    ifc_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else IFC_PATH
    db_arg  = Path(sys.argv[2]) if len(sys.argv) > 2 else BSOS_DB
    run_report(ifc_arg, db_arg)
