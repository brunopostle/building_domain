"""Section 16.4 — Pattern critique evaluation test.

Qualitative regression check: for each hand-authored case in
``tests/fixtures/pattern_critique_cases.json``, run the real
``bsos query <entity> --type pattern --include-proposed`` command against the
populated corpus and report which of the reviewer's ``expected_patterns`` were
surfaced and which were missing.

Relevance is subjective, so this test deliberately has **no pass/fail oracle on
pattern recall** — it prints a report a human reviewer reads after re-running
extraction (PROPOSAL.md §16.4). It does enforce two structural guarantees:

* the fixture is well-formed (every case has the required keys), and
* the fixture covers at least three distinct entity types (space, component,
  system) — the Phase-4 completeness criterion.

Gating: unlike the constraint/topology evaluations this needs no live LLM, only
a populated DB. The test skips with a clear message when the fixture has not yet
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
from bsos.persistence.models import PatternRow

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "pattern_critique_cases.json"
CORPUS_DB = REPO_ROOT / "bsos.db"

REQUIRED_TYPES = {"space", "component", "system"}

runner = CliRunner(mix_stderr=False)


def _load_cases():
    if not FIXTURE.exists():
        pytest.skip(
            f"Pattern critique fixture not authored yet: {FIXTURE}. "
            "Populate it with human-reviewed (entity, context, expected_patterns) "
            "cases before running the §16.4 pattern critique test."
        )
    return json.loads(FIXTURE.read_text())


def _corpus_engine():
    if not CORPUS_DB.exists():
        pytest.skip(
            f"No extracted corpus at {CORPUS_DB}; run the extraction pipeline first."
        )
    engine = create_db_engine(str(CORPUS_DB))
    with Session(engine) as s:
        if s.exec(select(PatternRow).limit(1)).first() is None:
            pytest.skip("Corpus DB has no Pattern rows; run extraction first.")
    return engine


def test_fixture_well_formed():
    """Every case carries the keys the §16.4 schema requires."""
    cases = _load_cases()
    assert isinstance(cases, list) and cases, "fixture must be a non-empty list"
    for i, case in enumerate(cases):
        assert "entity" in case, f"case {i} missing 'entity'"
        assert "context_description" in case, f"case {i} missing 'context_description'"
        assert isinstance(case.get("expected_patterns"), list) and case["expected_patterns"], (
            f"case {i} ({case.get('entity')}) needs a non-empty 'expected_patterns' list"
        )
        for ep in case["expected_patterns"]:
            assert ep.get("name"), f"case {i} has an expected pattern with no 'name'"
            assert ep.get("rationale"), (
                f"case {i} pattern '{ep.get('name')}' has no 'rationale'"
            )


def test_fixture_covers_three_entity_types():
    """The fixture spans space, component and system (Phase-4 completeness)."""
    cases = _load_cases()
    engine = _corpus_engine()
    found_types = set()
    unresolved = []
    with Session(engine) as s:
        for case in cases:
            row = resolve_entity(s, case["entity"])
            if row is None:
                unresolved.append(case["entity"])
            else:
                found_types.add(row.entity_type)
    assert not unresolved, f"fixture entities not in corpus: {unresolved}"
    missing = REQUIRED_TYPES - found_types
    assert not missing, (
        f"fixture must cover {sorted(REQUIRED_TYPES)}; missing {sorted(missing)} "
        f"(covered: {sorted(found_types)})"
    )


def test_pattern_critique_report(capsys):
    """Print found/missing expected patterns per case. Qualitative — never fails."""
    cases = _load_cases()
    _corpus_engine()  # skip early if no corpus

    lines = ["", "=== §16.4 Pattern critique report ==="]
    for case in cases:
        entity = case["entity"]
        result = runner.invoke(
            app,
            ["query", entity, "--type", "pattern", "--include-proposed",
             "--json", "--db", str(CORPUS_DB)],
        )
        returned = []
        if result.exit_code == 0:
            try:
                payload = json.loads(result.stdout)
                returned = [p["name"].lower() for p in payload.get("patterns", [])]
            except (json.JSONDecodeError, KeyError):
                pass

        expected = {ep["name"]: ep["name"].lower() in returned
                    for ep in case["expected_patterns"]}
        found = [n for n, ok in expected.items() if ok]
        missing = [n for n, ok in expected.items() if not ok]

        lines.append(f"\n[{entity}] {len(returned)} patterns returned")
        lines.append(f"  context: {case['context_description']}")
        lines.append(f"  FOUND  ({len(found)}): {found}")
        lines.append(f"  MISSING({len(missing)}): {missing}")

    report = "\n".join(lines)
    with capsys.disabled():
        print(report)
    # Intentionally no assertion on found/missing: relevance is a human judgement.
