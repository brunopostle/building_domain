"""Section 16.5 — Exception reasoning (coverage) test.

PROPOSAL.md §16.5 frames this as a *coverage check*, not a live-LLM oracle:
for a set of accepted assertions known to have documented exceptions, confirm
that querying the entity returns non-empty exception text alongside the base
assertion. Automated scoring of the exception *content* is not applicable —
relevance is a human judgement — so the only quantitative gate is that the
adversarial-validation pass (Pass 11) is actually populating
``Assertion.exceptions`` rather than leaving the lists empty.

Each hand-authored case in ``tests/fixtures/exception_cases.json`` names a
canonical building entity whose rules are well known to admit exceptions
(e.g. a window normally requires glazing, *except* louvred/solid-panel
windows). A case is *covered* when the corpus holds at least one accepted
assertion for that entity carrying a non-empty ``exceptions`` list (narrowed to
the case ``predicate`` when one is given).

Pass threshold (§16.5): ≥ 80% of fixture cases covered. Empty exception lists on
entities known to have exceptions indicate Pass 11 is not running, the
confidence threshold is filtering valid findings, or the adversarial model lacks
domain knowledge for that assertion.

Gating: like the §16.4 pattern critique this needs no live LLM — only a
populated corpus. The test skips with a clear message when the fixture has not
been authored or when no extracted corpus is available.
"""
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner
from sqlmodel import Session, select

from bsos.cli.main import app
from bsos.cli.query import resolve_entity
from bsos.persistence.database import create_db_engine
from bsos.persistence.models import AssertionRow

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "exception_cases.json"
CORPUS_DB = REPO_ROOT / "bsos.db"

REQUIRED_TYPES = {"space", "component", "system"}
COVERAGE_TARGET = 0.80

runner = CliRunner(mix_stderr=False)


def _load_cases():
    if not FIXTURE.exists():
        pytest.skip(
            f"Exception fixture not authored yet: {FIXTURE}. Populate it with "
            "human-reviewed (entity, expected_exception) cases for canonical "
            "rules that admit exceptions before running the §16.5 test."
        )
    return json.loads(FIXTURE.read_text())


def _corpus_engine():
    if not CORPUS_DB.exists():
        pytest.skip(
            f"No extracted corpus at {CORPUS_DB}; run the extraction pipeline first."
        )
    engine = create_db_engine(str(CORPUS_DB))
    with Session(engine) as s:
        if s.exec(select(AssertionRow).limit(1)).first() is None:
            pytest.skip("Corpus DB has no Assertion rows; run extraction first.")
    return engine


def test_fixture_well_formed():
    """Every case carries the keys the §16.5 schema requires."""
    cases = _load_cases()
    assert isinstance(cases, list) and cases, "fixture must be a non-empty list"
    for i, case in enumerate(cases):
        assert case.get("entity"), f"case {i} missing 'entity'"
        assert case.get("entity_type"), f"case {i} ({case.get('entity')}) missing 'entity_type'"
        assert case.get("expected_exception"), (
            f"case {i} ({case.get('entity')}) needs a non-empty 'expected_exception'"
        )


def test_fixture_covers_three_entity_types():
    """The fixture spans space, component and system (Phase-4 completeness)."""
    cases = _load_cases()
    declared = {c["entity_type"] for c in cases}
    missing = REQUIRED_TYPES - declared
    assert not missing, (
        f"fixture must cover {sorted(REQUIRED_TYPES)}; missing {sorted(missing)} "
        f"(covered: {sorted(declared)})"
    )


def _exception_assertions(session: Session, entity_row, predicate: str | None):
    """Accepted assertions touching ``entity_row`` that carry a non-empty exceptions list."""
    rows = session.exec(
        select(AssertionRow).where(
            (AssertionRow.subject_id == entity_row.id)
            | (AssertionRow.object_id == entity_row.id),
            AssertionRow.status == "accepted",
        )
    ).all()
    out = []
    for r in rows:
        if predicate and r.predicate != predicate:
            continue
        try:
            exceptions = json.loads(r.exceptions) if r.exceptions else []
        except json.JSONDecodeError:
            exceptions = []
        if exceptions:
            out.append(r)
    return out


def test_exception_coverage(capsys):
    """>= 80% of fixture entities have at least one recorded exception."""
    cases = _load_cases()
    engine = _corpus_engine()

    unresolved: list[str] = []
    covered: list[str] = []
    empty: list[str] = []

    with Session(engine) as s:
        for case in cases:
            entity = case["entity"]
            row = resolve_entity(s, entity)
            if row is None:
                unresolved.append(entity)
                continue
            hits = _exception_assertions(s, row, case.get("predicate"))
            (covered if hits else empty).append(entity)

    assert not unresolved, f"fixture entities not in corpus: {unresolved}"

    total = len(cases)
    coverage = len(covered) / total if total else 0.0

    report = [
        "",
        "=== §16.5 Exception coverage report ===",
        f"  covered: {len(covered)}/{total} ({coverage:.0%})",
        f"  EMPTY (no exceptions recorded): {empty}",
    ]
    with capsys.disabled():
        print("\n".join(report))

    assert coverage >= COVERAGE_TARGET, (
        f"Exception coverage {coverage:.0%} < {COVERAGE_TARGET:.0%}\n"
        f"Entities with no recorded exceptions: {empty}\n"
        "Pass 11 adversarial validation may not be populating Assertion.exceptions."
    )
