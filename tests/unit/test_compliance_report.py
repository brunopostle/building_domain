"""Unit tests for the pure helpers in scripts/ifc_compliance_report.py.

These cover the free-text → checkable-object mapping for constraints, the
spatial object → space-usage reverse map, and the conservative antipattern
trigger logic. The IFC/DB-driven parts (run_report) are exercised manually
against _test.ifc and are not unit-tested here.
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


# ── classify_constraint ──────────────────────────────────────────────────────

@pytest.mark.parametrize("rule,expected", [
    ("Kitchen must have a waterproof floor surface", "Waterproof Flooring"),
    ("Floor must have water-resistant covering", "Waterproof Flooring"),
    ("Floor finish must provide slip resistance of 0.5 COF", "Anti-slip Flooring"),
    ("must have a drainage path for wastewater", "Drainage System"),
    ("must have a water supply for flushing", "Plumbing System"),
    ("Kitchen must have a ventilation system or operable window",
     "Ventilation System or Operable Window"),
    ("must have ventilation to outside or mechanical exhaust",
     "Ventilation System or Operable Window"),
    ("Living room must have at least one window to the exterior", "Windows"),
    ("shop front glazing system that is laminated or tempered glass", "Windows"),
    ("must have a clear entrance width of at least 900 mm", "Doors"),
    ("must have a means of egress to a public way", "Doors"),
    ("external wall must have insulation", "Insulation"),
])
def test_classify_constraint_must(rule, expected):
    assert cr.classify_constraint(rule, "must") == expected


@pytest.mark.parametrize("rule", [
    "must have a minimum clear width of at least 36 inches",
    "must have a ceiling height of at least 7 feet 6 inches",
    "tread depth and riser height must be within a safe ratio",
    "must have a fire-resistance rating of 1 hour",
    "must have a dedicated 20-amp small-appliance branch circuit",
])
def test_classify_constraint_dimensional_is_unchecked(rule):
    assert cr.classify_constraint(rule, "must") is None


def test_classify_constraint_must_not_is_never_checked():
    # Prohibitions are geometric/code here and are not reduced to presence tests.
    rule = "must not have electrical outlets within 300mm of a sink basin"
    assert cr.classify_constraint(rule, "must_not") is None


def test_weatherproof_envelope_not_misread_as_waterproof_floor():
    # "water ingress" without "floor" must not map to Waterproof Flooring.
    rule = "must have a weatherproof envelope that prevents water ingress"
    assert cr.classify_constraint(rule, "must") != "Waterproof Flooring"


# ── object_to_usage ──────────────────────────────────────────────────────────

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


# ── antipattern_triggered ────────────────────────────────────────────────────

def test_antipattern_triggers_only_when_signal_present():
    signals = {"ventilation": True, "drainage": False,
               "window": False, "flooring": False}
    assert cr.antipattern_triggered(
        "Unvented or Poorly Vented Cooking Zone", "", signals) == "ventilation"


def test_antipattern_not_triggered_when_signal_absent():
    signals = {"ventilation": False, "drainage": False,
               "window": False, "flooring": False}
    assert cr.antipattern_triggered(
        "Unvented or Poorly Vented Cooking Zone", "", signals) is None


def test_antipattern_requires_topic_keyword():
    # All signals true, but the antipattern is about an ergonomic issue with
    # no detectable physical signal — must not be flagged.
    signals = {"ventilation": True, "drainage": True,
               "window": True, "flooring": True}
    assert cr.antipattern_triggered(
        "Microwave Over Range — Ergonomic Hazard", "", signals) is None
