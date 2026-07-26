"""Regression test for `bsos curate import-apl` (building_domain-tt0).

apl_patterns.json's higher_patterns/lower_patterns entries carry book slug ids
(e.g. "small-public-squares"), not PatternRow ids. import-apl must not write
those slugs into PatternRow.related_pattern_ids — that field is reserved for
real PatternRow ids, resolved later by pass10a from related_pattern_names.
"""
import json

from typer.testing import CliRunner
from sqlmodel import Session, select

from bsos.cli.main import app
from bsos.persistence.database import create_db_engine
from bsos.persistence.models import PatternRow

runner = CliRunner(mix_stderr=False)


def _init_db(tmp_path, name="t.db"):
    db = tmp_path / name
    runner.invoke(app, ["init", "--db", str(db), "--no-gitignore"])
    return str(db)


def test_import_apl_does_not_write_foreign_slug_ids(tmp_path):
    apl_path = tmp_path / "apl_patterns.json"
    apl_path.write_text(json.dumps([
        {
            "id": "small-public-squares",
            "name": "SMALL PUBLIC SQUARES",
            "section": "Towns",
            "problem": "problem text",
            "solution": "solution text",
            "confidence": "**",
            "higher_patterns": [{"id": "old-age-cottage", "name": "OLD AGE COTTAGE"}],
            "lower_patterns": [],
        },
        {
            "id": "old-age-cottage",
            "name": "OLD AGE COTTAGE",
            "section": "Towns",
            "problem": "problem text",
            "solution": "solution text",
            "confidence": "*",
            "higher_patterns": [],
            "lower_patterns": [{"id": "small-public-squares", "name": "SMALL PUBLIC SQUARES"}],
        },
    ]))

    db = _init_db(tmp_path)
    result = runner.invoke(app, ["curate", "import-apl", str(apl_path), "--db", db])
    assert result.exit_code == 0, result.output

    engine = create_db_engine(db)
    with Session(engine) as s:
        rows = {r.name: r for r in s.exec(select(PatternRow)).all()}

    assert len(rows) == 2
    for row in rows.values():
        assert json.loads(row.related_pattern_ids) == []
        assert json.loads(row.related_pattern_names) != []
