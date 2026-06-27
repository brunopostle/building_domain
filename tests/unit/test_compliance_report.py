"""Unit tests for the report-specific helpers in
scripts/ifc_compliance_report.py.

The deterministic check engine (constraint classification, evaluation,
antipatterns) now lives in bsos.validation and is tested in test_validation.py.
What remains here is the report's spatial-object → space-usage vocabulary used
for the adjacency checks.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "ifc_compliance_report", ROOT / "scripts" / "ifc_compliance_report.py"
)
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)


# ── footprint area from triangulated mesh ────────────────────────────────────

import math


class _FakeGeom:
    """Minimal stand-in for an ifcopenshell triangulation (flat verts/faces)."""
    def __init__(self, verts, faces):
        self.verts = verts
        self.faces = faces


def _square_floor(side: float, angle: float = 0.0) -> _FakeGeom:
    """A horizontal square floor of `side`, rotated `angle` rad about Z,
    as two upward-facing (CCW-from-above) triangles."""
    pts = [(0, 0), (side, 0), (side, side), (0, side)]
    ca, sa = math.cos(angle), math.sin(angle)
    verts = []
    for x, y in pts:
        verts += [x * ca - y * sa, x * sa + y * ca, 0.0]
    return _FakeGeom(verts, [0, 1, 2, 0, 2, 3])


def test_footprint_area_of_axis_aligned_square():
    assert cr._footprint_area(_square_floor(3.0)) == pytest.approx(9.0)


def test_footprint_area_is_rotation_invariant():
    # A room whose plan is not aligned to X/Y must report its true area, not an
    # inflated bounding-box footprint.
    assert cr._footprint_area(_square_floor(3.0, math.radians(30))) == pytest.approx(9.0)


def test_footprint_area_ignores_downward_faces():
    # A floor face (CCW, upward) plus its reversed copy (downward) — only the
    # upward face counts, so the footprint is the floor area, not double.
    g = _square_floor(2.0)
    g.faces = list(g.faces) + [2, 1, 0, 3, 2, 0]  # reversed winding = downward
    assert cr._footprint_area(g) == pytest.approx(4.0)


@pytest.mark.parametrize("name,usage", [
    ("Kitchen", "kitchen"),
    ("Living Room", "living"),
    ("Hallway", "circulation"),
    ("Corridor", "circulation"),
    ("Staircase", "stair"),
    ("Toilet / WC", "toilet"),
    ("Retail Unit", "retail"),
])
def test_object_to_usage_known(name, usage):
    assert cr.object_to_usage(name) == usage


@pytest.mark.parametrize("name", ["Countertop", "Wall", "Basement",
                                  "Fire Extinguisher", "Coffee Table"])
def test_object_to_usage_unknown(name):
    assert cr.object_to_usage(name) is None


def test_mep_presence_keys_are_known_system_objects():
    # The report's ifc-side system map must use the shared vocabulary, or
    # evaluate() will silently treat a system as an unknown object.
    from bsos import validation
    assert set(cr.MEP_PRESENCE) <= validation.SYSTEM_OBJECTS


def test_space_local_systems_are_known_mep_systems():
    # Every space-local system must also have a building-wide mapping, so the
    # two presence paths stay in sync.
    assert set(cr.SYSTEM_SPACE_FIXTURES) <= set(cr.MEP_PRESENCE)


# ── Per-space MEP presence (building_domain-d91) ─────────────────────────────
# Minimal duck-typed stand-ins for ifcopenshell entities: build_facts and
# count_bounded_by_type only touch .BoundedBy, .RelatedBuildingElement, .is_a()
# and ifc.by_type().

class _FakeElem:
    def __init__(self, cls):
        self._cls = cls

    def is_a(self, other=None):
        return self._cls if other is None else self._cls == other


class _FakeRel:
    def __init__(self, elem):
        self.RelatedBuildingElement = elem


class _FakeSpace:
    def __init__(self, *bounding_classes):
        self.Name = "S"
        self.BoundedBy = [_FakeRel(_FakeElem(c)) for c in bounding_classes]

    def id(self):
        return 1


class _FakeIfc:
    """ifc.by_type(cls) returns model-wide elements of that class."""
    def __init__(self, **counts):
        self._by_type = {cls: [_FakeElem(cls)] * n for cls, n in counts.items()}

    def by_type(self, cls):
        return self._by_type.get(cls, [])


@pytest.fixture(autouse=True)
def _clear_mep_cache():
    cr._mep_cache.clear()
    yield
    cr._mep_cache.clear()


def test_drainage_requires_fixture_in_this_space():
    # A sanitary terminal exists in the model but does NOT bound this space:
    # the old building-wide check passed drainage everywhere — now it must not.
    ifc = _FakeIfc(IfcSanitaryTerminal=1)
    no_fixture = _FakeSpace("IfcWall", "IfcWindow")
    assert "Drainage System" not in cr.build_facts(ifc, no_fixture)["systems_present"]


def test_drainage_passes_when_fixture_bounds_the_space():
    ifc = _FakeIfc(IfcSanitaryTerminal=1)
    with_fixture = _FakeSpace("IfcWall", "IfcSanitaryTerminal")
    assert "Drainage System" in cr.build_facts(ifc, with_fixture)["systems_present"]


def test_distribution_systems_stay_building_wide():
    # Electrical/lighting are not space-local; a model-wide element counts even
    # if it does not bound the space.
    ifc = _FakeIfc(IfcLightFixture=1)
    space = _FakeSpace("IfcWall")
    systems = cr.build_facts(ifc, space)["systems_present"]
    assert "Lighting System" in systems
    assert "Electrical System" in systems
