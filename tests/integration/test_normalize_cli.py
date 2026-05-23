"""Integration tests for bsos normalize command and Pass 10 extract integration."""
import json
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest
from sqlmodel import Session, select
from typer.testing import CliRunner

from bsos.cli.main import app
from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import (
    AbstractionNodeRow,
    AssertionRow,
    EmbeddingRow,
    EntityRow,
    PassProgressRow,
    PredicateMappingRow,
)

runner = CliRunner(mix_stderr=False)
NOW = datetime.now(timezone.utc)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_db(tmp_path):
    db = tmp_path / "bsos.db"
    result = runner.invoke(app, ["init", "--db", str(db), "--no-gitignore"])
    assert result.exit_code == 0, result.output
    return str(db)


def _engine(db_path: str):
    return create_db_engine(db_path)


def _add_entity(engine, name: str, status: str = "accepted") -> str:
    eid = str(uuid.uuid4())
    with Session(engine) as s:
        s.add(EntityRow(
            id=eid, name=name, entity_type="component", status=status,
            source_model="test", created_at=NOW,
        ))
        s.commit()
    return eid


def _add_assertion(engine, subject_id: str, object_id: str, predicate: str = "requires") -> str:
    aid = str(uuid.uuid4())
    with Session(engine) as s:
        s.add(AssertionRow(
            id=aid, subject_id=subject_id, predicate=predicate, object_id=object_id,
            subject_type="component", object_type="system",
            source_model="test", created_at=NOW,
            confidence=0.9, status="proposed", knowledge_origin="physical",
        ))
        s.commit()
    return aid


def _mark_pass_complete(engine, pass_number: str, embedding_model: str = "all-mpnet-base-v2") -> None:
    with Session(engine) as s:
        s.add(PassProgressRow(
            pass_number=pass_number,
            entity_id="__global__",
            model=embedding_model,
            completed_at=NOW,
            status="completed",
        ))
        s.commit()


# ---------------------------------------------------------------------------
# Fake embedder patch — monkeypatched into the normalization modules
# ---------------------------------------------------------------------------

def _identity_embedder(texts):
    """Returns distinct unit vectors by hashing text content."""
    result = []
    for t in texts:
        seed = abs(hash(t)) % (2**31)
        rng = np.random.default_rng(seed)
        v = rng.random(8).astype(np.float32) + 0.001
        result.append(v / float(np.linalg.norm(v)))
    return np.array(result, dtype=np.float32)


# ---------------------------------------------------------------------------
# bsos normalize — basic flag tests (using pre-completed passes to avoid LLM)
# ---------------------------------------------------------------------------

class TestNormalizeCommand:
    def test_no_assertions_exits_with_error(self, tmp_path):
        db = _init_db(tmp_path)
        runner.invoke(app, ["config", "set", "default_llm_model", "test-model", "--db", db])

        result = runner.invoke(app, ["normalize", "--db", db])
        assert result.exit_code != 0
        assert "No assertions" in result.output or "No assertions" in (result.stderr or "")

    def test_no_model_exits_with_error(self, tmp_path):
        db = _init_db(tmp_path)
        eng = _engine(db)
        e1 = _add_entity(eng, "roof")
        _add_assertion(eng, e1, e1)

        result = runner.invoke(app, ["normalize", "--db", db])
        assert result.exit_code != 0

    def test_all_passes_already_completed_reports_skip(self, tmp_path):
        db = _init_db(tmp_path)
        eng = _engine(db)
        e1 = _add_entity(eng, "roof")
        _add_assertion(eng, e1, e1)
        _mark_pass_complete(eng, "10a")
        _mark_pass_complete(eng, "10b")
        _mark_pass_complete(eng, "10c")

        result = runner.invoke(app, ["normalize", "--models", "test-model", "--db", db])
        assert result.exit_code == 0
        assert "already completed" in result.output

    def test_reembed_deletes_embedding_rows(self, tmp_path):
        db = _init_db(tmp_path)
        eng = _engine(db)
        e1 = _add_entity(eng, "roof")
        _add_assertion(eng, e1, e1)

        # Pre-populate some embedding rows and completed pass records.
        with Session(eng) as s:
            s.add(EmbeddingRow(
                item_type="assertion", item_id=str(uuid.uuid4()),
                model="all-mpnet-base-v2", dim=8,
                content_hash="abc", vector=b"\x00" * 32,
            ))
            s.commit()
        _mark_pass_complete(eng, "10a")
        _mark_pass_complete(eng, "10b")
        _mark_pass_complete(eng, "10c")

        # After reembed, embedding rows should be gone and passes should re-run.
        # We pre-mark passes complete again after deletion — just test deletion here.
        # Re-mark them so the run completes without needing a real LLM.
        result = runner.invoke(app, ["normalize", "--reembed", "--models", "test-model", "--db", db])
        # Passes will be re-run after deletion; since we deleted progress rows,
        # 10c will need a real LLM and may fail — check deletion happened at least.
        with Session(eng) as s:
            rows = s.exec(select(EmbeddingRow)).all()
            assert len(rows) == 0

    def test_reembed_writes_calibration_config(self, tmp_path):
        db = _init_db(tmp_path)
        eng = _engine(db)
        e1 = _add_entity(eng, "roof")
        _add_assertion(eng, e1, e1)
        _mark_pass_complete(eng, "10a")
        _mark_pass_complete(eng, "10b")
        _mark_pass_complete(eng, "10c")

        runner.invoke(app, ["normalize", "--reembed", "--models", "test-model", "--db", db])
        # Config write happens before passes run, so it's set regardless of LLM availability.
        from bsos.config import get_config
        with Session(eng) as s:
            val = get_config(s, "embedding_model_at_last_calibration")
        assert val == "all-mpnet-base-v2"

    def test_reembed_warns_about_thresholds(self, tmp_path):
        db = _init_db(tmp_path)
        eng = _engine(db)
        e1 = _add_entity(eng, "roof")
        _add_assertion(eng, e1, e1)
        _mark_pass_complete(eng, "10a")
        _mark_pass_complete(eng, "10b")
        _mark_pass_complete(eng, "10c")

        result = runner.invoke(app, ["normalize", "--reembed", "--models", "test-model", "--db", db])
        assert "Warning" in result.output or "threshold" in result.output.lower()

    def test_dry_run_makes_no_changes(self, tmp_path):
        db = _init_db(tmp_path)
        eng = _engine(db)
        e1 = _add_entity(eng, "roof")
        _add_assertion(eng, e1, e1, predicate="demands")  # non-core predicate

        result = runner.invoke(
            app, ["normalize", "--dry-run", "--models", "test-model", "--db", db]
        )
        assert result.exit_code == 0
        assert "dry-run" in result.output.lower() or "dry_run" in result.output.lower()

        # No mappings or nodes should have been written.
        with Session(eng) as s:
            assert s.exec(select(PredicateMappingRow)).all() == []
            assert s.exec(select(AbstractionNodeRow)).all() == []
            assert s.get(PassProgressRow, ("10a", "__global__", "all-mpnet-base-v2")) is None


# ---------------------------------------------------------------------------
# bsos status — normalization passes section
# ---------------------------------------------------------------------------

class TestStatusNormalizationPasses:
    def test_status_shows_no_normalization_passes(self, tmp_path):
        db = _init_db(tmp_path)
        result = runner.invoke(app, ["status", "--db", db])
        assert result.exit_code == 0
        # "Normalization passes" line should not appear when none complete.
        assert "Normalization passes" not in result.output

    def test_status_shows_completed_normalization_passes(self, tmp_path):
        db = _init_db(tmp_path)
        eng = _engine(db)
        _mark_pass_complete(eng, "10a")
        _mark_pass_complete(eng, "10b")

        result = runner.invoke(app, ["status", "--db", db])
        assert result.exit_code == 0
        assert "Normalization passes completed: 10a, 10b" in result.output

    def test_status_json_includes_normalization_passes(self, tmp_path):
        db = _init_db(tmp_path)
        eng = _engine(db)
        _mark_pass_complete(eng, "10c")

        result = runner.invoke(app, ["status", "--json", "--db", db])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "normalization_passes_completed" in data
        assert "10c" in data["normalization_passes_completed"]

    def test_status_normalization_passes_not_confused_with_model_passes(self, tmp_path):
        """10a/10b/10c must not appear in passes_completed under any model key."""
        db = _init_db(tmp_path)
        eng = _engine(db)
        _mark_pass_complete(eng, "10a")
        _mark_pass_complete(eng, "10b")
        _mark_pass_complete(eng, "10c")

        result = runner.invoke(app, ["status", "--json", "--db", db])
        data = json.loads(result.output)
        for model_passes in data.get("passes_completed", {}).values():
            assert "10a" not in model_passes
            assert "10b" not in model_passes
            assert "10c" not in model_passes


# ---------------------------------------------------------------------------
# bsos extract --passes 10 integration
# ---------------------------------------------------------------------------

class TestExtractPass10:
    def test_passes_flag_10_runs_normalize_passes(self, tmp_path):
        """bsos extract --passes 10 runs normalize (all sub-passes complete → skipped output)."""
        db = _init_db(tmp_path)
        eng = _engine(db)
        e1 = _add_entity(eng, "roof")
        _add_assertion(eng, e1, e1)
        # Pre-mark all sub-passes complete so normalize exits cleanly without LLM.
        _mark_pass_complete(eng, "10a")
        _mark_pass_complete(eng, "10b")
        _mark_pass_complete(eng, "10c")

        result = runner.invoke(app, [
            "extract", "--passes", "10", "--models", "test-model", "--db", db,
        ])
        assert result.exit_code == 0, result.output
        assert "already completed" in result.output

    def test_passes_flag_10_skips_llm_passes(self, tmp_path):
        """bsos extract --passes 10 does not attempt to create an LLM provider for passes 1-9."""
        db = _init_db(tmp_path)
        eng = _engine(db)
        e1 = _add_entity(eng, "roof")
        _add_assertion(eng, e1, e1)
        _mark_pass_complete(eng, "10a")
        _mark_pass_complete(eng, "10b")
        _mark_pass_complete(eng, "10c")

        # If provider creation were attempted, it would raise OpenAIError (no creds).
        # A successful exit proves provider creation was skipped.
        result = runner.invoke(app, [
            "extract", "--passes", "10", "--models", "definitely-not-a-real-model", "--db", db,
        ])
        assert result.exit_code == 0, result.output
