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
