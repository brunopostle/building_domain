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


# ── classify_dimensional_constraint ──────────────────────────────────────────

@pytest.mark.parametrize("rule,kind,threshold", [
    ("must have a minimum floor area of at least 0.9 m x 1.2 m",
     "min_floor_area", 1.08),
    ("Bedroom must have a minimum floor area of 7 m²",
     "min_floor_area", 7.0),
    ("Plant room must have a minimum floor area of 4.5 square metres",
     "min_floor_area", 4.5),
    ("must have a ceiling height of at least 7 feet 6 inches (2286 mm)",
     "min_ceiling_height", 2.286),
    ("Living room must have a minimum ceiling height of at least 2.1 m (7 ft)",
     "min_ceiling_height", 2.1),
    ("Staircase must have a minimum headroom clearance of 2.0 m",
     "min_ceiling_height", 2.0),
])
def test_classify_dimensional_constraint(rule, kind, threshold):
    result = v.classify_dimensional_constraint(rule, "must")
    assert result is not None
    assert result[0] == kind
    assert result[1] == pytest.approx(threshold)


def test_ceiling_rule_mentioning_floor_area_classifies_as_height():
    # "...in at least 50% of the floor area" must not be read as an area check.
    rule = ("Living room must have a minimum ceiling height of at least 2.1 m "
            "(7 ft) in at least 50% of the floor area")
    assert v.classify_dimensional_constraint(rule, "must") == ("min_ceiling_height", 2.1)


@pytest.mark.parametrize("rule", [
    "must have a minimum clear width of at least 36 inches (914 mm)",  # width, not area/height
    "tread depth and riser height must be within a safe ratio",
    "must have a dedicated 20-amp small-appliance branch circuit",
])
def test_classify_dimensional_constraint_non_area_height_is_none(rule):
    assert v.classify_dimensional_constraint(rule, "must") is None


def test_classify_dimensional_constraint_skips_must_not():
    rule = "must not exceed a maximum floor area of 50 m²"
    assert v.classify_dimensional_constraint(rule, "must_not") is None


# ── evaluate_dimensional ──────────────────────────────────────────────────────

def test_evaluate_dimensional_floor_area_pass_fail_unchecked():
    assert v.evaluate_dimensional("min_floor_area", 1.08,
                                  {"floor_area_m2": 12.2})[0] == v.PASS
    assert v.evaluate_dimensional("min_floor_area", 10.0,
                                  {"floor_area_m2": 4.0})[0] == v.FAIL
    assert v.evaluate_dimensional("min_floor_area", 1.08, {})[0] == v.UNCHECKED


def test_evaluate_dimensional_ceiling_height_pass_fail_unchecked():
    assert v.evaluate_dimensional("min_ceiling_height", 2.1,
                                  {"ceiling_height_m": 3.0})[0] == v.PASS
    assert v.evaluate_dimensional("min_ceiling_height", 2.4,
                                  {"ceiling_height_m": 2.1})[0] == v.FAIL
    assert v.evaluate_dimensional("min_ceiling_height", 2.1, {})[0] == v.UNCHECKED


def test_evaluate_dimensional_at_threshold_passes():
    # Exact-equal measurement must pass despite mesh round-off.
    assert v.evaluate_dimensional("min_ceiling_height", 2.4,
                                  {"ceiling_height_m": 2.4})[0] == v.PASS


def test_validate_constraints_routes_dimensional():
    constraints = [
        {"rule": "must have a minimum floor area of at least 0.9 m x 1.2 m",
         "constraint_type": "must", "confidence": 0.95},
    ]
    [result] = v.validate_constraints(constraints, {"floor_area_m2": 12.2})
    assert result["check_object"] == "min_floor_area"
    assert result["status"] == v.PASS


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


# ── classify_pattern ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,problem,solution", [
    ("Light on Two Sides", "Rooms lit from only one side suffer from uneven "
     "light distribution.", "Place windows on two sides of the room."),
    ("Daylight From Two Sides for Room Ambiance", "Rooms lit from only one "
     "side suffer harsh shadows.", "Add a second window wall."),
    ("Light from Above or Two Sides to Reduce Glare",
     "Windows on one side create glare.", "Add clerestory glazing or a "
     "second window wall."),
])
def test_classify_pattern_light_on_two_sides(name, problem, solution):
    assert v.classify_pattern(name, problem, solution) == "Light on Two Sides"


def test_classify_pattern_unrelated_two_sides_wording_not_matched():
    # "two sides" without any light/daylight/window context should not match.
    assert v.classify_pattern(
        "Accessible From Two Sides", "A corridor needs entry points.",
        "Place doors at both ends.") is None


def test_classify_pattern_unmapped_pattern_is_none():
    assert v.classify_pattern(
        "Window-Connected Individual Workstations",
        "Workers in deep interior zones lack daylight.",
        "Arrange workstations near windows.") is None


def test_evaluate_light_on_two_sides():
    assert v.evaluate("Light on Two Sides", {})[0] == v.UNCHECKED
    assert v.evaluate("Light on Two Sides", {"window_wall_count": 0})[0] == v.FAIL
    assert v.evaluate("Light on Two Sides", {"window_wall_count": 1})[0] == v.FAIL
    assert v.evaluate("Light on Two Sides", {"window_wall_count": 2})[0] == v.PASS
    assert v.evaluate("Light on Two Sides", {"window_wall_count": 3})[0] == v.PASS


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
