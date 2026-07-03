#!/usr/bin/env python3
"""IFC BOQ (bill-of-quantities) sanity check.

Cross-references the bill of quantities implied by a model's actual geometry
-- a per-IFC-class element count and total volume derived from triangulated
geometry, standing in for `ifc_quantify`/`ifc_cost` (this script talks to
ifcopenshell directly rather than the external `ifcmcp` package, same as
ifc_compliance_report.py) -- against BSOS `requires`/`depends_on` assertions
for the matched component entity.

Each present IFC class is resolved to a bsos 'component' entity via the same
semantic-search resolver introduced for the compliance report
(building_domain-l5w.1), so this works against any element type rather than a
fixed list. A high-confidence requirement for a material or component that
shows up nowhere in the model -- neither as another present element type nor
as a material name anywhere -- is flagged as a likely BOQ omission (e.g. a
Foundation quantified in the model but no Damp Proof Course material or
element anywhere).

Only requirements whose bsos object is itself a 'material' or 'component'
entity are checked: activities and systems (e.g. "Soil Investigation",
"Structural Engineering Design") are services/processes, not something a BOQ
line item would ever represent, so checking them would just be noise.

Usage:
    python scripts/ifc_boq_check.py [path/to/model.ifc] [path/to/bsos.db]
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element
import ifcopenshell.util.shape

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ifc_compliance_report as compliance  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IFC_PATH = ROOT / "_test.ifc"
BSOS_DB  = ROOT / "bsos.db"

# Only worth flagging requirements confident enough to act on.
OMISSION_MIN_CONFIDENCE = 0.85

# Component names cleaned from IFC class names (e.g. 'IfcFooting' -> 'Footing')
# match their bsos entity almost exactly (calibrated 2026-07-02/03: ~1.0 for
# Footing/Wall/Slab/Beam/Column/Roof/Window/Door; ~0.64 for the vaguer
# 'Covering'), so use a stricter threshold than the space resolver's 0.4 to
# avoid matching a novel/unmapped IFC class to an unrelated bsos entity.
COMPONENT_MATCH_MIN_SCORE = 0.6

_GEOM_SETTINGS = ifcopenshell.geom.settings()

# Only these bsos entity types are ever plausible BOQ line items.
CHECKABLE_OBJECT_TYPES = {"material", "component"}


def _clean_class_name(ifc_class: str) -> str:
    """'IfcFooting' -> 'Footing', 'IfcSanitaryTerminal' -> 'Sanitary Terminal'."""
    name = ifc_class[3:] if ifc_class.startswith("Ifc") else ifc_class
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).strip()


def collect_model_materials(ifc) -> set[str]:
    """Lowercased material names used anywhere in the model."""
    mats: set[str] = set()
    for element in ifc.by_type("IfcElement"):
        compliance._collect_layer_materials(element, mats)
    return mats


def element_quantities(ifc) -> dict[str, dict]:
    """Per-IFC-class {count, volume_m3, has_volume} derived from geometry.

    Stands in for `ifc_quantify`/`ifc_cost`. volume_m3 stays 0 (has_volume
    False) when geometry can't be triangulated (e.g. 2D annotations) or
    Representation is absent.
    """
    out: dict[str, dict] = defaultdict(lambda: {"count": 0, "volume_m3": 0.0, "has_volume": False})
    for element in ifc.by_type("IfcElement"):
        entry = out[element.is_a()]
        entry["count"] += 1
        if getattr(element, "Representation", None) is not None:
            try:
                shape = ifcopenshell.geom.create_shape(_GEOM_SETTINGS, element)
                entry["volume_m3"] += ifcopenshell.util.shape.get_volume(shape.geometry)
                entry["has_volume"] = True
            except Exception:
                pass
    return out


def requirement_evidenced(obj: str, present_components: set[str], model_materials: set[str]) -> bool:
    """True if a required entity shows up as another present component or as a
    material used anywhere in the model (substring match, case-insensitive --
    a best-effort heuristic, same imprecision tradeoff as embedding search)."""
    obj_lower = obj.lower()
    if obj_lower in present_components:
        return True
    return any(obj_lower in mat or mat in obj_lower for mat in model_materials)


def run_boq_check(ifc_path: Path = IFC_PATH, db_path: Path = BSOS_DB, _embedder=None) -> list[dict]:
    ifc = ifcopenshell.open(str(ifc_path))
    quantities = element_quantities(ifc)
    model_materials = collect_model_materials(ifc)

    engine = compliance.create_db_engine(str(db_path))
    resolved: dict[str, str] = {}
    with compliance.Session(engine) as session:
        for cls in quantities:
            entity = compliance.semantic_match_entity(
                session, _clean_class_name(cls), entity_type="component",
                min_score=COMPONENT_MATCH_MIN_SCORE, _embedder=_embedder)
            if entity:
                resolved[cls] = entity

    present_components_lower = {e.lower() for e in resolved.values()}

    rows: list[dict] = []
    print(f"\n{'='*compliance.W}")
    print("  IFC BOQ Sanity Check")
    print(f"  Model : {Path(ifc_path).name}")
    print(f"{'='*compliance.W}\n")

    for cls in sorted(resolved):
        entity = resolved[cls]
        qty = quantities[cls]
        vol_str = f", {qty['volume_m3']:.1f} m³" if qty["has_volume"] else ""
        print(f"▶  {entity}  ({cls}: {qty['count']} element(s){vol_str})")

        reqs = compliance.get_requirements(db_path, entity)
        checkable = [
            r for r in reqs
            if r[2] >= OMISSION_MIN_CONFIDENCE
            and compliance.get_entity_type(db_path, r[1]) in CHECKABLE_OBJECT_TYPES
        ]
        if not checkable:
            print("   ·  no high-confidence material/component requirements to check")
            print()
            continue

        for predicate, obj, confidence, rationale, conds, appl in checkable:
            evidenced = requirement_evidenced(obj, present_components_lower, model_materials)
            status = "PASS" if evidenced else "FAIL"
            symbol = "✓" if evidenced else "✗"
            detail = ("found elsewhere in the model" if evidenced else
                      "not found as a material or element anywhere in the model "
                      "-- possible BOQ omission")
            print(f"   {symbol} [{confidence:.0%}] {predicate} {obj}")
            print(f"         {detail}")
            rows.append({"ifc_class": cls, "entity": entity, "predicate": predicate,
                         "object": obj, "confidence": confidence, "status": status,
                         "detail": detail})
        print()

    total = len(rows)
    failed = sum(1 for r in rows if r["status"] == "FAIL")
    print(f"{'='*compliance.W}")
    print(f"  Checks: {total}   ✓ PASS {total - failed}   ✗ possible omissions {failed}")
    print(f"{'='*compliance.W}\n")

    if failed:
        print("POSSIBLE BOQ OMISSIONS")
        print(f"{'─'*compliance.W}")
        for r in rows:
            if r["status"] == "FAIL":
                print(f"  ✗  {r['entity']} ({r['ifc_class']}) — {r['object']}  [{r['confidence']:.0%}]")
        print()

    return rows


if __name__ == "__main__":
    ifc_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else IFC_PATH
    db_arg  = Path(sys.argv[2]) if len(sys.argv) > 2 else BSOS_DB
    run_boq_check(ifc_arg, db_arg)
