"""Unit tests for scripts/ifc_boq_check.py (building_domain-l5w.7).

The full run_boq_check() flow (real ifcopenshell geometry + real embedding
model) was verified manually against _test.ifc / bsos.db during development --
see the beads issue notes. What's covered here is the two pure-logic helpers:
IFC class name cleaning and the material/component evidence check.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "ifc_boq_check", ROOT / "scripts" / "ifc_boq_check.py"
)
boq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(boq)


@pytest.mark.parametrize("ifc_class,expected", [
    ("IfcFooting", "Footing"),
    ("IfcSanitaryTerminal", "Sanitary Terminal"),
    ("IfcWall", "Wall"),
    ("IfcAirTerminalBox", "Air Terminal Box"),
])
def test_clean_class_name(ifc_class, expected):
    assert boq._clean_class_name(ifc_class) == expected


def test_clean_class_name_without_ifc_prefix():
    # Defensive: is_a() always returns an Ifc-prefixed name in practice, but
    # the slice guard means unexpected input degrades gracefully rather than
    # mangling the first three characters of an unrelated string.
    assert boq._clean_class_name("Wall") == "Wall"


def test_requirement_evidenced_by_present_component():
    assert boq.requirement_evidenced("Footing", {"footing", "wall"}, set())


def test_requirement_evidenced_by_model_material():
    assert boq.requirement_evidenced(
        "Concrete", set(), {"concrete c30/37"})


def test_requirement_evidenced_case_insensitive():
    assert boq.requirement_evidenced("CONCRETE", set(), {"concrete c30/37"})


def test_requirement_not_evidenced():
    assert not boq.requirement_evidenced(
        "Reinforcing Steel", {"footing", "wall"}, {"concrete c30/37"})


def test_requirement_evidenced_reverse_substring():
    # A material name that is a substring of the (longer) requirement name
    # also counts -- e.g. model material 'timber' vs requirement 'Timber Beam'.
    assert boq.requirement_evidenced("Timber Beam", set(), {"timber"})
