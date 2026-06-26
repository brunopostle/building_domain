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

    floor_materials : list[str]   lower-case covering material names
    window_count    : int         operable windows bounding the space
    door_count      : int         doors bounding the space
    systems_present : list[str]   BSOS system names modelled ("Drainage System")
    has_insulation  : bool        any wall assembly has an insulation layer
    adjacent_usages : list[str]   space usages sharing a boundary

The deterministic engine only covers rule shapes it has patterns for. Anything
it cannot map returns status UNCHECKED with reason ``no_deterministic_matcher``
— a typed result a future LLM-assisted pass can fill in without changing the
contract.
"""
from __future__ import annotations

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


def validate_constraints(constraints: list[dict], facts: dict) -> list[dict]:
    """Validate a list of constraint dicts against facts.

    Each constraint dict needs at least ``rule`` and ``constraint_type`` and may
    carry ``confidence``. Returns one verdict dict per constraint.
    """
    out: list[dict] = []
    for c in constraints:
        obj = classify_constraint(c["rule"], c["constraint_type"])
        if obj is None:
            status, detail = UNCHECKED, NO_MATCHER
        else:
            status, detail = evaluate(obj, facts)
        out.append({
            "rule": c["rule"],
            "constraint_type": c["constraint_type"],
            "confidence": c.get("confidence"),
            "check_object": obj,
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
