"""Unit tests for the deterministic validation engine (bsos.validation).

Pure functions: constraint-rule classification, fact-based evaluation, the
whole-constraint-list helper, and conservative antipattern triggering.
"""
import pytest

from bsos import validation as v


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
    assert v.classify_constraint(rule, "must") == expected


@pytest.mark.parametrize("rule", [
    "must have a minimum clear width of at least 36 inches",
    "must have a ceiling height of at least 7 feet 6 inches",
    "tread depth and riser height must be within a safe ratio",
    "must have a fire-resistance rating of 1 hour",
    "must have a dedicated 20-amp small-appliance branch circuit",
])
def test_classify_constraint_dimensional_is_unchecked(rule):
    assert v.classify_constraint(rule, "must") is None


def test_classify_constraint_must_not_is_never_checked():
    rule = "must not have electrical outlets within 300mm of a sink basin"
    assert v.classify_constraint(rule, "must_not") is None


def test_weatherproof_envelope_not_misread_as_waterproof_floor():
    rule = "must have a weatherproof envelope that prevents water ingress"
    assert v.classify_constraint(rule, "must") != "Waterproof Flooring"


# ── evaluate ──────────────────────────────────────────────────────────────────

def test_evaluate_waterproof_floor_pass_and_fail():
    status, _ = v.evaluate("Waterproof Flooring", {"floor_materials": ["tiles"]})
    assert status == v.PASS
    status, detail = v.evaluate("Waterproof Flooring", {"floor_materials": ["carpet"]})
    assert status == v.FAIL and "carpet" in detail


def test_evaluate_system_presence():
    assert v.evaluate("Drainage System",
                      {"systems_present": ["Drainage System"]})[0] == v.PASS
    assert v.evaluate("Drainage System",
                      {"systems_present": []})[0] == v.FAIL


def test_evaluate_absent_fact_is_unchecked_not_fail():
    # window_count absent -> UNCHECKED; present-but-zero -> FAIL.
    assert v.evaluate("Windows", {})[0] == v.UNCHECKED
    assert v.evaluate("Windows", {"window_count": 0})[0] == v.FAIL
    assert v.evaluate("Windows", {"window_count": 3})[0] == v.PASS


def test_evaluate_ventilation_or_window():
    assert v.evaluate("Ventilation System or Operable Window",
                      {"systems_present": ["Ventilation System"]})[0] == v.PASS
    assert v.evaluate("Ventilation System or Operable Window",
                      {"window_count": 1, "systems_present": []})[0] == v.PASS
    assert v.evaluate("Ventilation System or Operable Window",
                      {"window_count": 0, "systems_present": []})[0] == v.FAIL


def test_evaluate_unknown_object_is_typed_unchecked():
    status, detail = v.evaluate("Some Unmodellable Thing", {})
    assert status == v.UNCHECKED and detail == v.NO_MATCHER


def test_evaluate_non_geometric_requirement_unchecked():
    status, _ = v.evaluate("Ceiling", {"floor_materials": ["tiles"]})
    assert status == v.UNCHECKED


# ── validate_constraints ──────────────────────────────────────────────────────

def test_validate_constraints_mixed():
    constraints = [
        {"rule": "Kitchen must have a waterproof floor surface",
         "constraint_type": "must", "confidence": 0.99},
        {"rule": "Kitchen must have a 20-amp branch circuit",
         "constraint_type": "must", "confidence": 0.9},
        {"rule": "Kitchen must not have outlets above the sink",
         "constraint_type": "must_not", "confidence": 0.8},
    ]
    facts = {"floor_materials": ["tiles"]}
    results = v.validate_constraints(constraints, facts)
    assert [r["status"] for r in results] == [v.PASS, v.UNCHECKED, v.UNCHECKED]
    assert results[0]["check_object"] == "Waterproof Flooring"
    # unmatched / prohibition rules carry the typed reason
    assert results[1]["detail"] == v.NO_MATCHER
    assert results[2]["check_object"] is None


# ── antipatterns ──────────────────────────────────────────────────────────────

def test_antipattern_triggers_only_when_signal_present():
    facts = {"systems_present": [], "window_count": 0}  # ventilation signal True
    signals = v.antipattern_signals(facts)
    assert v.antipattern_triggered(
        "Unvented or Poorly Vented Cooking Zone", "", signals) == "ventilation"


def test_antipattern_not_triggered_when_signal_absent():
    facts = {"systems_present": ["Ventilation System"], "window_count": 2}
    signals = v.antipattern_signals(facts)
    assert v.antipattern_triggered(
        "Unvented or Poorly Vented Cooking Zone", "", signals) is None


def test_antipattern_requires_topic_keyword():
    facts = {"systems_present": [], "window_count": 0, "floor_materials": []}
    signals = v.antipattern_signals(facts)
    assert v.antipattern_triggered(
        "Microwave Over Range — Ergonomic Hazard", "", signals) is None


def test_antipattern_accepts_conditions_list():
    facts = {"systems_present": [], "window_count": 0}
    signals = v.antipattern_signals(facts)
    topic = v.antipattern_triggered(
        "Cooking Zone", ["Range hood absent", "no exhaust ducting"], signals)
    assert topic == "ventilation"
