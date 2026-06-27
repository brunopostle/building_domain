"""Section 16.2 — Constraint reasoning evaluation test.

Planted-violation harness: seeds canonical building constraints alongside
hand-authored *violating* and *valid* assertions, runs the real
``bsos validate constraints <entity>`` command against a live LLM, and measures:

* detection rate  = violations correctly flagged / total synthetic violations
* false-positive rate = valid assertions incorrectly flagged / total valid cases

Passing threshold (PROPOSAL.md §16.2): detection rate >= 0.80. The false-positive
rate is tracked and held to a generous upper bound so a model that flags
everything cannot pass on recall alone.

Gate: set BSOS_E2E=1 to run (live LLM + scratch DB). The model defaults to
claude-haiku-4-5-20251001; override with BSOS_E2E_MODEL.
"""
import os
import pytest
from datetime import datetime, timezone
from typer.testing import CliRunner
from sqlmodel import Session

from bsos.cli.main import app
from bsos.persistence.database import create_db_engine
from bsos.persistence.models import AssertionRow, ConstraintRow, EntityRow

pytestmark = pytest.mark.skipif(
    not os.environ.get("BSOS_E2E"),
    reason="Set BSOS_E2E=1 to run end-to-end evaluation tests",
)

runner = CliRunner(mix_stderr=False)
NOW = datetime.now(timezone.utc)
MODEL = os.environ.get("BSOS_E2E_MODEL", "claude-haiku-4-5-20251001")

# Passing thresholds (PROPOSAL.md §16.2).
DETECTION_THRESHOLD = 0.80
MAX_FALSE_POSITIVE_RATE = 0.20


# ---------------------------------------------------------------------------
# Fixture cases — canonical building constraints with a clearly violating and a
# clearly compliant assertion each. These are deliberately unambiguous so that a
# competent LLM should classify them correctly; ambiguous edge cases belong in
# the conflict / exception tests, not the constraint-reasoning oracle.
# ---------------------------------------------------------------------------

CASES = [
    {
        "entity": "roof",
        "type": "must",
        "rule": "A roof must have a drainage path that removes rainwater from its surface.",
        "violating": "has no gutters, outlets or falls; rainwater ponds on the surface with nowhere to drain",
        "valid": "is laid to falls and discharges to gutters and downpipes",
    },
    {
        "entity": "staircase",
        "type": "must",
        "rule": "A staircase must maintain at least 2.0m of clear headroom along its pitch line.",
        "violating": "has only 1.6m of clear headroom under the landing above",
        "valid": "provides 2.1m of clear headroom over the full flight",
    },
    {
        "entity": "habitable room",
        "type": "must",
        "rule": "Every habitable room must have at least one means of escape in case of fire.",
        "violating": "is an inner room with no door, openable window or other escape route",
        "valid": "has an escape window and a doorway onto a protected corridor",
    },
    {
        "entity": "load-bearing wall",
        "type": "must_not",
        "rule": "A load-bearing wall must not be removed without a substitute structural support.",
        "violating": "was demolished to form an opening with no beam or column installed to carry the load above",
        "valid": "has an opening formed beneath a steel beam sized to carry the load above",
    },
    {
        "entity": "fire compartment wall",
        "type": "must",
        "rule": "A fire compartment wall must be continuous from the floor to the underside of the floor or roof above.",
        "violating": "stops at the suspended ceiling, leaving an open gap in the void above it",
        "valid": "extends fully to the underside of the structural slab above",
    },
    {
        "entity": "external masonry wall",
        "type": "must",
        "rule": "An external masonry wall must have a damp-proof course at least 150mm above external ground level.",
        "violating": "has its damp-proof course buried below the external ground level",
        "valid": "has its damp-proof course set 150mm above the finished ground level",
    },
    {
        "entity": "foundation",
        "type": "must",
        "rule": "A foundation must bear on adequate strata below the frost line.",
        "violating": "is founded 100mm down on soft made ground above the frost depth",
        "valid": "is taken down to firm undisturbed clay below the frost depth",
    },
    {
        "entity": "internal bathroom",
        "type": "must",
        "rule": "A bathroom without an openable window must have mechanical extract ventilation.",
        "violating": "is an internal room with no window and no extract fan",
        "valid": "is an internal room fitted with a ducted mechanical extract fan",
    },
    {
        "entity": "balcony",
        "type": "must",
        "rule": "A balcony edge above a 600mm drop must be protected by guarding at least 1100mm high.",
        "violating": "has only a 500mm high rail along an open edge above a 3m drop",
        "valid": "has a 1100mm high balustrade along its open edge",
    },
    {
        "entity": "floor beam",
        "type": "must_not",
        "rule": "A floor beam's deflection under design load must not exceed span/360.",
        "violating": "deflects by span/120 under its design load",
        "valid": "is sized so deflection is limited to span/400 under design load",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    return name.lower().replace(" ", "-").replace("/", "-")


def _init_db(tmp_path):
    db = tmp_path / "constraint_eval.db"
    runner.invoke(app, ["init", "--db", str(db), "--no-gitignore"])
    return str(db)


def _seed(db_path, variant: str):
    """Seed one entity per case carrying its constraint and the chosen assertion.

    ``variant`` is "violating" or "valid" and selects which hand-authored
    assertion is attached to each entity.
    """
    eng = create_db_engine(db_path)
    with Session(eng) as s:
        # Shared object node so every assertion has a valid object_id target.
        s.add(EntityRow(
            id="e-state", name="building state", entity_type="property",
            status="accepted", source_model="eval_harness", created_at=NOW,
        ))
        for case in CASES:
            eid = "e-" + _slug(case["entity"])
            s.add(EntityRow(
                id=eid, name=case["entity"], entity_type="concept",
                status="accepted", source_model="eval_harness", created_at=NOW,
            ))
            s.add(ConstraintRow(
                id="c-" + _slug(case["entity"]),
                subject_id=eid,
                rule=case["rule"],
                constraint_type=case["type"],
                status="accepted",
                confidence=1.0,
                knowledge_origin="architectural",
                source_model="eval_harness",
                created_at=NOW,
            ))
            s.add(AssertionRow(
                id="a-" + _slug(case["entity"]),
                subject_id=eid,
                predicate="has_property",
                object_id="e-state",
                subject_type="concept",
                object_type="property",
                status="accepted",
                confidence=0.95,
                knowledge_origin="architectural",
                source_model="eval_harness",
                created_at=NOW,
                rationale=case[variant],
            ))
        s.commit()


def _flagged(db, entity: str) -> bool:
    """Run `bsos validate constraints <entity>`; True if a violation was flagged."""
    result = runner.invoke(
        app, ["validate", "constraints", entity, "--db", db, "--model", MODEL]
    )
    # The command raises Exit(1) (and prints "violation(s) found") only when a
    # constraint is violated; clean runs ("No constraint violations found") exit 0.
    return result.exit_code != 0 and "violation" in result.output.lower()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_constraint_detection_rate(tmp_path):
    """>= 80% of planted violations are correctly flagged."""
    db = _init_db(tmp_path)
    _seed(db, "violating")

    detected = [c["entity"] for c in CASES if _flagged(db, c["entity"])]
    missed = [c["entity"] for c in CASES if c["entity"] not in detected]
    rate = len(detected) / len(CASES)

    assert rate >= DETECTION_THRESHOLD, (
        f"Detection rate {rate:.0%} < {DETECTION_THRESHOLD:.0%}\n"
        f"Missed violations: {missed}"
    )


def test_constraint_false_positive_rate(tmp_path):
    """Compliant assertions are not over-flagged as violations."""
    db = _init_db(tmp_path)
    _seed(db, "valid")

    false_positives = [c["entity"] for c in CASES if _flagged(db, c["entity"])]
    rate = len(false_positives) / len(CASES)

    assert rate <= MAX_FALSE_POSITIVE_RATE, (
        f"False-positive rate {rate:.0%} > {MAX_FALSE_POSITIVE_RATE:.0%}\n"
        f"Wrongly flagged: {false_positives}"
    )
