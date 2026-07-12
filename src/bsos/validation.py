"""Deterministic constraint / requirement validation engine.

Pure functions that turn a BSOS knowledge item — a prose constraint rule, a
requirement object name, or an antipattern — plus a dict of concrete *facts*
extracted from a model into a structured pass / fail / unchecked verdict.

There is no IFC or database dependency here: the caller supplies the facts.
That keeps the knowledge layer (BSOS) and the model layer (the `ifc` MCP
server) cleanly separated — an agent extracts facts from the model via the
`ifc` tools and passes them in; the compliance-report script extracts the same
facts from an IfcOpenShell model. Both share this one engine, so the MCP
`validate_element` tool and the report can never disagree.

Facts schema (every key optional; a fact that is absent yields UNCHECKED for
checks that need it, which is distinct from a fact that is present-but-empty
yielding FAIL):

    floor_materials    : list[str]  lower-case covering material names
    window_count       : int        operable windows bounding the space
    window_wall_count  : int        distinct walls carrying a window bounding
                                     the space (light-from-N-sides signal)
    door_count         : int        doors bounding the space
    systems_present    : list[str]  BSOS system names modelled ("Drainage System")
    has_insulation     : bool       any wall assembly has an insulation layer
    adjacent_usages    : list[str]  space usages sharing a boundary
    floor_area_m2      : float      true (plan-projected) floor area in m²
    ceiling_height_m   : float      floor-to-ceiling height in m

The deterministic engine only covers rule shapes it has patterns for. Anything
it cannot map returns status UNCHECKED with reason ``no_deterministic_matcher``
— a typed result a future LLM-assisted pass can fill in without changing the
contract.
"""
from __future__ import annotations

import re

PASS, FAIL, UNCHECKED = "pass", "fail", "unchecked"

# ── Material quality sets ────────────────────────────────────────────────────
SLIP_RESISTANT = {"tiles", "concrete", "stone", "vinyl", "screed",
                  "terrazzo", "ceramic", "epoxy", "resin", "linoleum"}
WATERPROOF     = {"tiles", "concrete", "stone", "vinyl", "ceramic",
                  "epoxy", "resin", "membrane", "linoleum"}

FLOORING_QUALITY: dict[str, set[str]] = {
    "Anti-slip Flooring": SLIP_RESISTANT,
    "Waterproof Flooring": WATERPROOF,
}

# Canonical BSOS system names whose check is a presence test against
# facts["systems_present"]. The ifc-side mapping of each name to concrete IFC
# classes lives in the compliance-report client (MEP_PRESENCE); this engine
# only needs the vocabulary.
SYSTEM_OBJECTS = frozenset({
    "Ventilation System", "HVAC System", "Heating System", "Electrical System",
    "Lighting System", "Drainage System", "Rough-in Plumbing", "Plumbing System",
    "Fire Suppression System", "Fire Protection System", "Security System",
    "Structural Frame", "Foundation",
})

# Requirement objects that cannot be verified from geometry/materials alone.
UNCHECKABLE_OBJECTS = {
    "Countertop Material", "Refrigeration Unit", "Nosing Profile",
    "Door Hardware", "Stringer", "Landing", "Ceiling",
    "Countertop", "Cabinetry",
}

NO_MATCHER = "no_deterministic_matcher"


# ── Free-text constraint rule → checkable object ─────────────────────────────

def classify_constraint(rule: str, constraint_type: str) -> str | None:
    """Map a constraint rule to a check object, or None if not checkable.

    Only `must` rules whose required element maps to a physical signal are
    checkable; `must_not` prohibitions here are geometric/code (clearances,
    projections, sill heights) and do not reduce to a presence test.
    """
    if constraint_type == "must_not":
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


# ── Dimensional constraint rule → (check kind, SI threshold) ─────────────────
# Rules that classify_constraint cannot reduce to a presence test but that *do*
# reduce to a numeric threshold checkable against space geometry: minimum floor
# area and minimum floor-to-ceiling height. Other dimensional rules (clear
# corridor width, tread/riser ratios, sill heights, nosing projections) need
# geometry the space record does not expose and remain unchecked.

# Length tokens give a value + metric unit. Imperial values ("36 inches",
# "7 feet") are deliberately ignored: every rule in the corpus restates the
# metric equivalent, usually in a parenthetical, so taking only metric tokens
# avoids unit-conversion guesswork. ``m`` requires a following non-letter so it
# never swallows the "m" of "mm"/"m²".
_LEN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mm|cm|m)(?![a-z²])", re.I)
# Area as an explicit "N m²" / "N square metres" token …
_AREA_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:m²|m2|sq\.?\s*m|square\s+met(?:re|er)s?)", re.I)
# … or as a "0.9 m x 1.2 m" plan-dimension product.
_DIM_PRODUCT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*m\s*[x×]\s*(\d+(?:\.\d+)?)\s*m", re.I)


def _metric_lengths_m(text: str) -> list[float]:
    """All metric length tokens in ``text``, normalised to metres."""
    scale = {"mm": 0.001, "cm": 0.01, "m": 1.0}
    return [float(v) * scale[u.lower()] for v, u in _LEN_RE.findall(text)]


def classify_dimensional_constraint(
        rule: str, constraint_type: str) -> tuple[str, float] | None:
    """Map a constraint rule to (kind, SI threshold), or None if not numeric.

    kind is one of ``min_floor_area`` (m²) or ``min_ceiling_height`` (m). Only
    ``must`` minimums are returned; ``must_not`` maxima and non-dimensional
    rules yield None so the caller can fall back to the unchecked bucket.
    """
    if constraint_type == "must_not":
        return None
    r = rule.lower()

    if "floor area" in r or ("area" in r and ("m²" in r or "m2" in r)):
        m = _DIM_PRODUCT_RE.search(rule)
        if m:
            return "min_floor_area", round(float(m.group(1)) * float(m.group(2)), 3)
        m = _AREA_RE.search(rule)
        if m:
            return "min_floor_area", float(m.group(1))

    if ("ceiling height" in r or "headroom" in r) and (
            "least" in r or "minimum" in r or "min " in r):
        lengths = _metric_lengths_m(rule)
        if lengths:
            return "min_ceiling_height", max(lengths)

    return None


# ── Free-text pattern name/problem/solution → checkable object ───────────────

def classify_pattern(name: str, problem: str, solution: str) -> str | None:
    """Map an Alexander-style pattern to a check object, or None if not
    mechanically checkable.

    Mirrors classify_constraint's keyword approach: most patterns are
    qualitative design guidance ("Window-Connected Individual Workstations",
    "Stair as Light Monitor") with no single geometric signal to test, so only
    patterns with an unambiguous fact-derivable claim are mapped. Currently
    covers "light on N sides" wording, the one shape build_facts already
    exposes a distinct signal for (window_wall_count).
    """
    text = f"{name} {problem} {solution}".lower()
    if "two sides" in text and ("light" in text or "daylight" in text or "window" in text):
        return "Light on Two Sides"
    return None


# ── Verdict ──────────────────────────────────────────────────────────────────

def evaluate(check_object: str, facts: dict) -> tuple[str, str]:
    """Return (status, detail) for a check object against supplied facts.

    status is one of PASS / FAIL / UNCHECKED. A required fact that is absent
    (None) yields UNCHECKED; present-but-empty yields FAIL.
    """
    obj = check_object
    floor_mats = set(facts.get("floor_materials") or [])
    systems = set(facts.get("systems_present") or [])

    if obj == "Flooring":
        if floor_mats:
            return PASS, f"floor material present: {', '.join(sorted(floor_mats))}"
        return FAIL, "no floor covering found"

    if obj in FLOORING_QUALITY:
        approved = FLOORING_QUALITY[obj]
        good = floor_mats & approved
        if good:
            return PASS, f"approved material: {', '.join(sorted(good))}"
        if floor_mats:
            return FAIL, (
                f"floor is {', '.join(sorted(floor_mats))} "
                f"— not in approved set {{{', '.join(sorted(approved))}}}"
            )
        return FAIL, "no floor covering found"

    if obj == "Windows":
        n = facts.get("window_count")
        if n is None:
            return UNCHECKED, "window_count not supplied"
        return (PASS, f"{n} window(s)") if n else (FAIL, "no windows found")

    if obj == "Light on Two Sides":
        n = facts.get("window_wall_count")
        if n is None:
            return UNCHECKED, "window_wall_count not supplied"
        if n >= 2:
            return PASS, f"windows on {n} distinct wall(s)"
        if n == 1:
            return FAIL, "windows on only 1 wall — not lit from two sides"
        return FAIL, "no windows found"

    if obj in ("Doors", "Entrance Door", "External Door"):
        n = facts.get("door_count")
        if n is None:
            return UNCHECKED, "door_count not supplied"
        return (PASS, f"{n} door(s)") if n else (FAIL, "no doors found")

    if obj == "Ventilation System or Operable Window":
        win = facts.get("window_count") or 0
        if "Ventilation System" in systems:
            return PASS, "ventilation system present"
        if win:
            return PASS, f"{win} operable window(s) in space"
        return FAIL, "no ventilation system and no window in space"

    if obj == "Insulation":
        hi = facts.get("has_insulation")
        if hi is None:
            return UNCHECKED, "has_insulation not supplied"
        return ((PASS, "insulation layer found in wall assemblies") if hi
                else (FAIL, "no insulation layer in any wall assembly"))

    if obj in SYSTEM_OBJECTS:
        if "systems_present" not in facts:
            return UNCHECKED, "systems_present not supplied"
        return ((PASS, "system elements present in model") if obj in systems
                else (FAIL, "no MEP elements of this type found in model"))

    if obj in UNCHECKABLE_OBJECTS:
        return UNCHECKED, "requires non-geometric IFC data"

    return UNCHECKED, NO_MATCHER


def evaluate_dimensional(kind: str, threshold: float, facts: dict) -> tuple[str, str]:
    """Compare a measured space dimension against a minimum threshold.

    ``kind`` / ``threshold`` come from :func:`classify_dimensional_constraint`.
    A missing measurement yields UNCHECKED (distinct from a measured value that
    falls short, which is FAIL). A small tolerance absorbs meshing round-off.
    """
    if kind == "min_floor_area":
        area = facts.get("floor_area_m2")
        if area is None:
            return UNCHECKED, "floor_area_m2 not supplied"
        if area + 1e-6 >= threshold:
            return PASS, f"floor area {area:.1f} m² ≥ required {threshold:.2f} m²"
        return FAIL, f"floor area {area:.1f} m² < required {threshold:.2f} m²"

    if kind == "min_ceiling_height":
        h = facts.get("ceiling_height_m")
        if h is None:
            return UNCHECKED, "ceiling_height_m not supplied"
        if h + 1e-6 >= threshold:
            return PASS, f"ceiling height {h:.2f} m ≥ required {threshold:.2f} m"
        return FAIL, f"ceiling height {h:.2f} m < required {threshold:.2f} m"

    return UNCHECKED, NO_MATCHER


def validate_constraints(constraints: list[dict], facts: dict) -> list[dict]:
    """Validate a list of constraint dicts against facts.

    Each constraint dict needs at least ``rule`` and ``constraint_type`` and may
    carry ``confidence``. Returns one verdict dict per constraint.
    """
    out: list[dict] = []
    for c in constraints:
        obj = classify_constraint(c["rule"], c["constraint_type"])
        if obj is not None:
            check_object = obj
            status, detail = evaluate(obj, facts)
        else:
            dim = classify_dimensional_constraint(c["rule"], c["constraint_type"])
            if dim is not None:
                kind, threshold = dim
                check_object = kind
                status, detail = evaluate_dimensional(kind, threshold, facts)
            else:
                check_object = None
                status, detail = UNCHECKED, NO_MATCHER
        out.append({
            "rule": c["rule"],
            "constraint_type": c["constraint_type"],
            "confidence": c.get("confidence"),
            "check_object": check_object,
            "status": status,
            "detail": detail,
        })
    return out


# ── Antipatterns: negative-signal detection ──────────────────────────────────
# Flag an antipattern only when its topic keyword co-occurs with a negative
# signal we can confirm from the facts, so a complete model yields no false
# positives.
ANTIPATTERN_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ventilation": ("unvented", "poorly vented", "no ventilation", "stale air",
                    "stuffy", "no exhaust", "no airflow"),
    "drainage":    ("no drainage", "no drain", "missing drainage", "no plumbing",
                    "standing water", "sewage backup"),
    "window":      ("no window", "windowless", "no daylight", "no natural light",
                    "no view", "no exterior wall"),
    "flooring":    ("no floor finish", "bare subfloor", "missing floor"),
}


def antipattern_signals(facts: dict) -> dict[str, bool]:
    """Negative signals that hold for these facts (True = failure condition)."""
    systems = set(facts.get("systems_present") or [])
    win = facts.get("window_count") or 0
    return {
        "ventilation": not ("Ventilation System" in systems or win),
        "drainage":    "Drainage System" not in systems,
        "window":      win == 0,
        "flooring":    not (facts.get("floor_materials") or []),
    }


def antipattern_triggered(name: str, conditions, signals: dict[str, bool]) -> str | None:
    """Return the triggering topic if a confirmable failure signal matches.

    ``conditions`` may be a string or a list of strings.
    """
    if isinstance(conditions, (list, tuple)):
        conditions = " ".join(conditions)
    text = (name + " " + (conditions or "")).lower()
    for topic, keywords in ANTIPATTERN_TOPIC_KEYWORDS.items():
        if signals.get(topic) and any(kw in text for kw in keywords):
            return topic
    return None
