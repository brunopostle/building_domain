"""Phase 3 integration tests: history, graph construction, and export/cache CLI.

Covers the items from building_domain-vej that are not already tested by the
dedicated pass10a/b/c, auto-promotion, conflict-detection, and abstraction-view
test files:

  6. bsos history — synthesized initial entry, chronological transitions, JSON
  7. Graph construction — node/edge counts, lazy subgraph, spatial edges,
     status filtering, save/load round-trip
  + Export and cache CLI smoke tests (from building_domain-3k6, building_domain-s6i)
"""
import json
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pytest
from sqlmodel import Session, select
from typer.testing import CliRunner

from bsos.cli.main import app
from bsos.cli.history import run_history
from bsos.graph import build_full_graph, build_lazy_subgraph, save_graph, load_graph
from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import (
    AbstractionNodeRow,
    AssertionRow,
    EntityRow,
    ForceRow,
    PatternRow,
    ProcessRelationRow,
    ProvenanceLogRow,
    SpatialRelationRow,
)

runner = CliRunner(mix_stderr=False)
NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _engine(tmp_path):
    db = tmp_path / "t.db"
    eng = create_db_engine(str(db))
    create_views(eng)
    return eng


def _init_db(tmp_path):
    db = tmp_path / "t.db"
    runner.invoke(app, ["init", "--db", str(db), "--no-gitignore"])
    return str(db)


def _entity(session, eid, name, entity_type="component", status="proposed"):
    row = EntityRow(id=eid, name=name, entity_type=entity_type,
                    status=status, source_model="test", created_at=NOW)
    session.add(row)
    return row


def _assertion(session, aid, subject_id, predicate, object_id,
               status="proposed", source_model="test", confidence=0.8):
    row = AssertionRow(
        id=aid, subject_id=subject_id, predicate=predicate, object_id=object_id,
        subject_type="component", object_type="component",
        confidence=confidence, knowledge_origin="engineering",
        status=status, source_model=source_model, created_at=NOW,
    )
    session.add(row)
    return row


def _provenance(session, pid, item_id, item_type, old_status, new_status, delta_seconds=0):
    row = ProvenanceLogRow(
        id=pid,
        item_id=item_id,
        item_type=item_type,
        old_status=old_status,
        new_status=new_status,
        changed_at=NOW + timedelta(seconds=delta_seconds),
        changed_by="bsos-test",
    )
    session.add(row)
    return row


# ---------------------------------------------------------------------------
# bsos history — run_history function tests
# ---------------------------------------------------------------------------

class TestRunHistory:
    @pytest.fixture
    def engine(self, tmp_path):
        return _engine(tmp_path)

    def test_initial_entry_appears_first(self, engine, capsys):
        with Session(engine) as s:
            _entity(s, "e1", "Roof")
            s.commit()

        with Session(engine) as s:
            run_history(s, "e1", json_out=False)

        out = capsys.readouterr().out
        assert "proposed" in out

    def test_initial_entry_has_no_old_status(self, engine):
        with Session(engine) as s:
            _entity(s, "e1", "Roof")
            s.commit()

        with Session(engine) as s:
            run_history(s, "e1", json_out=True)

        # captured via capsys is awkward — use direct call + capture
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with Session(engine) as s:
                run_history(s, "e1", json_out=True)
        transitions = json.loads(buf.getvalue())
        assert transitions[0]["old_status"] is None
        assert transitions[0]["new_status"] == "proposed"
        assert transitions[0]["label"] == "(initial)"

    def test_transitions_in_chronological_order(self, engine):
        with Session(engine) as s:
            _entity(s, "e1", "Wall")
            _provenance(s, "p1", "e1", "entity", "proposed", "accepted", delta_seconds=10)
            _provenance(s, "p2", "e1", "entity", "accepted", "conflicted", delta_seconds=20)
            s.commit()

        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with Session(engine) as s:
                run_history(s, "e1", json_out=True)
        transitions = json.loads(buf.getvalue())

        assert len(transitions) == 3   # initial + two
        assert transitions[0]["new_status"] == "proposed"
        assert transitions[1]["new_status"] == "accepted"
        assert transitions[2]["new_status"] == "conflicted"

    def test_no_provenance_log_just_initial(self, engine):
        with Session(engine) as s:
            _entity(s, "e1", "Floor")
            s.commit()

        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with Session(engine) as s:
                run_history(s, "e1", json_out=True)
        transitions = json.loads(buf.getvalue())
        assert len(transitions) == 1

    def test_item_not_found_exits(self, engine):
        import sys
        with pytest.raises(SystemExit):
            with Session(engine) as s:
                run_history(s, "nonexistent-uuid", json_out=False)

    def test_works_for_assertion(self, engine):
        with Session(engine) as s:
            _entity(s, "e1", "Wall")
            _entity(s, "e2", "Foundation")
            _assertion(s, "a1", "e1", "depends_on", "e2")
            _provenance(s, "p1", "a1", "assertion", "proposed", "accepted", delta_seconds=5)
            s.commit()

        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with Session(engine) as s:
                run_history(s, "a1", json_out=True)
        transitions = json.loads(buf.getvalue())
        assert transitions[0]["new_status"] == "proposed"
        assert transitions[1]["new_status"] == "accepted"


# ---------------------------------------------------------------------------
# bsos history CLI
# ---------------------------------------------------------------------------

class TestHistoryCLI:
    def test_cli_history_json(self, tmp_path):
        db = _init_db(tmp_path)
        eng = create_db_engine(db)
        with Session(eng) as s:
            _entity(s, "e1", "Roof")
            _provenance(s, "p1", "e1", "entity", "proposed", "accepted", delta_seconds=5)
            s.commit()

        result = runner.invoke(app, ["history", "--json", "e1", "--db", db])
        assert result.exit_code == 0
        transitions = json.loads(result.output)
        assert len(transitions) == 2
        assert transitions[0]["new_status"] == "proposed"
        assert transitions[1]["new_status"] == "accepted"

    def test_cli_history_text(self, tmp_path):
        db = _init_db(tmp_path)
        eng = create_db_engine(db)
        with Session(eng) as s:
            _entity(s, "e1", "Wall")
            s.commit()

        result = runner.invoke(app, ["history", "e1", "--db", db])
        assert result.exit_code == 0
        assert "proposed" in result.output

    def test_cli_history_not_found_exits(self, tmp_path):
        db = _init_db(tmp_path)
        result = runner.invoke(app, ["history", "no-such-id", "--db", db])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Graph construction — build_full_graph
# ---------------------------------------------------------------------------

class TestBuildFullGraph:
    @pytest.fixture
    def engine(self, tmp_path):
        return _engine(tmp_path)

    def test_entity_nodes_present(self, engine):
        with Session(engine) as s:
            _entity(s, "e1", "Roof")
            _entity(s, "e2", "Precipitation")
            s.commit()

        with Session(engine) as s:
            g = build_full_graph(s)

        assert "e1" in g
        assert "e2" in g
        assert g.nodes["e1"]["node_type"] == "entity"
        assert g.nodes["e1"]["name"] == "Roof"

    def test_assertion_edge_added(self, engine):
        with Session(engine) as s:
            _entity(s, "e1", "Roof")
            _entity(s, "e2", "Precipitation")
            _assertion(s, "a1", "e1", "protects_from", "e2")
            s.commit()

        with Session(engine) as s:
            g = build_full_graph(s)

        assert g.has_edge("e1", "e2")
        assert g["e1"]["e2"]["edge_type"] == "protects_from"
        assert g["e1"]["e2"]["edge_category"] == "assertion"

    def test_process_relation_precedes_edge(self, engine):
        with Session(engine) as s:
            _entity(s, "e1", "Formwork")
            _entity(s, "e2", "Concrete Pouring")
            s.add(ProcessRelationRow(
                id="pr1", predecessor_id="e1", successor_id="e2",
                hard_constraint=True, rationale="Must set form first",
                confidence=0.9, knowledge_origin="engineering",
                source_model="test", created_at=NOW,
            ))
            s.commit()

        with Session(engine) as s:
            g = build_full_graph(s)

        assert g.has_edge("e1", "e2")
        assert g["e1"]["e2"]["edge_type"] == "precedes"
        assert g["e1"]["e2"]["edge_category"] == "structural"

    def test_spatial_relation_edge(self, engine):
        with Session(engine) as s:
            _entity(s, "e1", "Room")
            _entity(s, "e2", "Corridor")
            s.add(SpatialRelationRow(
                id="sr1", subject_id="e1", relation="accessible_from",
                object_id="e2", confidence=0.85, knowledge_origin="architectural",
                source_model="test", created_at=NOW,
            ))
            s.commit()

        with Session(engine) as s:
            g = build_full_graph(s)

        assert g.has_edge("e1", "e2")
        assert g["e1"]["e2"]["edge_type"] == "accessible_from"
        assert g["e1"]["e2"]["edge_category"] == "spatial"

    def test_deprecated_assertions_excluded_by_default(self, engine):
        with Session(engine) as s:
            _entity(s, "e1", "Wall")
            _entity(s, "e2", "Foundation")
            _assertion(s, "a1", "e1", "depends_on", "e2", status="deprecated")
            s.commit()

        with Session(engine) as s:
            g = build_full_graph(s)

        # Entity nodes still present but the assertion edge should not be
        assert "e1" in g
        assert not g.has_edge("e1", "e2")

    def test_accepted_only_filter_excludes_proposed(self, engine):
        with Session(engine) as s:
            _entity(s, "e1", "Wall")
            _entity(s, "e2", "Foundation")
            _assertion(s, "a1", "e1", "depends_on", "e2", status="proposed")
            s.commit()

        with Session(engine) as s:
            g = build_full_graph(s, min_status="accepted")

        assert not g.has_edge("e1", "e2")

    def test_abstraction_node_aggregates_edge(self, engine):
        with Session(engine) as s:
            _entity(s, "e1", "Wall")
            _entity(s, "e2", "Roof")
            _assertion(s, "a1", "e1", "requires", "e2")
            s.add(AbstractionNodeRow(
                id="ab1", statement="Enclosure requires structure",
                child_ids=json.dumps(["a1"]),
                abstraction_rationale="Synthesized",
                confidence=0.8, status="proposed",
                source_model="test", created_at=NOW,
            ))
            s.commit()

        with Session(engine) as s:
            g = build_full_graph(s)

        assert "ab1" in g
        assert g.nodes["ab1"]["node_type"] == "abstraction_node"
        assert g.has_edge("ab1", "a1")
        assert g["ab1"]["a1"]["edge_type"] == "aggregates"

    def test_node_and_edge_counts(self, engine):
        with Session(engine) as s:
            for i in range(5):
                _entity(s, f"e{i}", f"Entity{i}")
            _assertion(s, "a1", "e0", "requires", "e1")
            _assertion(s, "a2", "e1", "depends_on", "e2")
            s.commit()

        with Session(engine) as s:
            g = build_full_graph(s)

        assert g.number_of_nodes() >= 5
        assert g.has_edge("e0", "e1")
        assert g.has_edge("e1", "e2")


# ---------------------------------------------------------------------------
# Graph construction — build_lazy_subgraph
# ---------------------------------------------------------------------------

class TestBuildLazySubgraph:
    @pytest.fixture
    def engine(self, tmp_path):
        return _engine(tmp_path)

    def test_lazy_subgraph_contains_queried_entity(self, engine):
        with Session(engine) as s:
            _entity(s, "e1", "Roof")
            _entity(s, "e2", "Membrane")
            _entity(s, "e3", "Unrelated")
            _assertion(s, "a1", "e1", "requires", "e2")
            s.commit()

        with Session(engine) as s:
            g = build_lazy_subgraph(s, "e1")

        assert "e1" in g
        assert "e2" in g
        assert "e3" not in g

    def test_lazy_subgraph_traverses_in_both_directions(self, engine):
        with Session(engine) as s:
            _entity(s, "e1", "Foundation")
            _entity(s, "e2", "Wall")
            _entity(s, "e3", "Roof")
            _assertion(s, "a1", "e2", "depends_on", "e1")
            _assertion(s, "a2", "e3", "depends_on", "e2")
            s.commit()

        with Session(engine) as s:
            g = build_lazy_subgraph(s, "e2")

        assert "e1" in g
        assert "e2" in g
        assert "e3" in g

    def test_lazy_subgraph_unknown_entity_returns_empty(self, engine):
        with Session(engine) as s:
            _entity(s, "e1", "Roof")
            s.commit()

        with Session(engine) as s:
            g = build_lazy_subgraph(s, "nonexistent-id")

        assert g.number_of_nodes() == 0


# ---------------------------------------------------------------------------
# Graph save/load round-trip
# ---------------------------------------------------------------------------

class TestGraphSaveLoad:
    @pytest.fixture
    def engine(self, tmp_path):
        return _engine(tmp_path)

    def test_save_and_load_preserves_graph(self, engine, tmp_path):
        with Session(engine) as s:
            _entity(s, "e1", "Roof")
            _entity(s, "e2", "Foundation")
            _assertion(s, "a1", "e1", "requires", "e2")
            s.commit()

        with Session(engine) as s:
            g = build_full_graph(s)

        from bsos.graph import get_schema_version
        schema_v = get_schema_version(engine)
        path = tmp_path / "graph.pkl"
        save_graph(g, path, schema_v)

        assert path.exists()
        assert (Path(str(path) + ".sha256")).exists()

        loaded = load_graph(path, engine)
        assert loaded.number_of_nodes() == g.number_of_nodes()
        assert loaded.number_of_edges() == g.number_of_edges()

    def test_load_raises_on_checksum_mismatch(self, engine, tmp_path):
        with Session(engine) as s:
            _entity(s, "e1", "Roof")
            s.commit()

        with Session(engine) as s:
            g = build_full_graph(s)

        from bsos.graph import get_schema_version
        path = tmp_path / "graph.pkl"
        save_graph(g, path, get_schema_version(engine))

        # corrupt the checksum
        sha_path = Path(str(path) + ".sha256")
        sha_path.write_text("deadbeef" * 8)

        with pytest.raises(ValueError, match="checksum"):
            load_graph(path, engine)


# ---------------------------------------------------------------------------
# Export CLI smoke tests
# ---------------------------------------------------------------------------

class TestExportCLI:
    def test_export_json_all_types(self, tmp_path):
        db = _init_db(tmp_path)
        result = runner.invoke(app, ["export", "--format", "json", "--db", db])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert set(data.keys()) >= {
            "entities", "assertions", "constraints", "patterns",
            "forces", "antipatterns", "spatial_relations",
        }

    def test_export_json_single_type(self, tmp_path):
        db = _init_db(tmp_path)
        eng = create_db_engine(db)
        with Session(eng) as s:
            _entity(s, "e1", "Roof")
            s.commit()

        result = runner.invoke(app, ["export", "--type", "entities", "--format", "json", "--db", db])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "entities" in data
        assert any(e["name"] == "Roof" for e in data["entities"])

    def test_export_csv_single_type(self, tmp_path):
        db = _init_db(tmp_path)
        eng = create_db_engine(db)
        with Session(eng) as s:
            _entity(s, "e1", "Wall")
            s.commit()

        result = runner.invoke(app, ["export", "--type", "entities", "--format", "csv", "--db", db])
        assert result.exit_code == 0
        assert "name" in result.output   # header row
        assert "Wall" in result.output

    def test_export_csv_multiple_types_errors(self, tmp_path):
        db = _init_db(tmp_path)
        result = runner.invoke(
            app, ["export", "--format", "csv", "--type", "entities", "--type", "assertions", "--db", db]
        )
        assert result.exit_code != 0

    def test_export_to_file(self, tmp_path):
        db = _init_db(tmp_path)
        out = str(tmp_path / "out.json")
        result = runner.invoke(app, ["export", "--format", "json", "--output", out, "--db", db])
        assert result.exit_code == 0
        assert Path(out).exists()
        data = json.loads(Path(out).read_text())
        assert "entities" in data

    def test_export_status_filter(self, tmp_path):
        db = _init_db(tmp_path)
        eng = create_db_engine(db)
        with Session(eng) as s:
            _entity(s, "e1", "Roof", status="accepted")
            _entity(s, "e2", "Wall", status="proposed")
            s.commit()

        result = runner.invoke(
            app, ["export", "--type", "entities", "--format", "json",
                  "--status", "accepted", "--db", db]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        names = [e["name"] for e in data["entities"]]
        assert "Roof" in names
        assert "Wall" not in names


# ---------------------------------------------------------------------------
# Import CLI tests
# ---------------------------------------------------------------------------

class TestImportCLI:
    def test_import_roundtrip_entities_and_assertions(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "dst").mkdir()
        src_db = _init_db(tmp_path / "src")
        dst_db = _init_db(tmp_path / "dst")
        eng = create_db_engine(src_db)
        with Session(eng) as s:
            _entity(s, "e1", "Roof", entity_type="component")
            _entity(s, "e2", "Wall", entity_type="component")
            _assertion(s, "a1", "e1", "requires", "e2")
            s.commit()

        snapshot = tmp_path / "snap.json"
        runner.invoke(app, ["export", "--format", "json", "--output", str(snapshot), "--db", src_db])

        result = runner.invoke(app, ["import", "--input", str(snapshot), "--db", dst_db])
        assert result.exit_code == 0, result.output
        assert "entities: 2 added" in result.output
        assert "assertions: 1 added" in result.output

        dst_eng = create_db_engine(dst_db)
        with Session(dst_eng) as s:
            from bsos.persistence.models import AssertionRow as AR
            rows = s.exec(select(AR)).all()
        assert len(rows) == 1
        assert rows[0].predicate == "requires"

    def test_import_skip_existing_by_default(self, tmp_path):
        db = _init_db(tmp_path)
        eng = create_db_engine(db)
        with Session(eng) as s:
            _entity(s, "e1", "Roof")
            s.commit()

        snapshot = tmp_path / "snap.json"
        runner.invoke(app, ["export", "--format", "json", "--output", str(snapshot), "--db", db])

        result = runner.invoke(app, ["import", "--input", str(snapshot), "--db", db])
        assert result.exit_code == 0
        assert "skipped" in result.output

    def test_import_replace_overwrites(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "dst").mkdir()
        src_db = _init_db(tmp_path / "src")
        dst_db = _init_db(tmp_path / "dst")
        eng_src = create_db_engine(src_db)
        eng_dst = create_db_engine(dst_db)
        with Session(eng_src) as s:
            _entity(s, "e1", "Roof", entity_type="component")
            s.commit()
        with Session(eng_dst) as s:
            _entity(s, "e1", "Roof", entity_type="space")  # different type
            s.commit()

        snapshot = tmp_path / "snap.json"
        runner.invoke(app, ["export", "--format", "json", "--output", str(snapshot), "--db", src_db])

        result = runner.invoke(app, ["import", "--input", str(snapshot), "--replace", "--db", dst_db])
        assert result.exit_code == 0
        with Session(eng_dst) as s:
            from bsos.persistence.models import EntityRow as ER
            row = s.get(ER, "e1")
        assert row.entity_type == "component"

    def test_import_stdin(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "dst").mkdir()
        src_db = _init_db(tmp_path / "src")
        dst_db = _init_db(tmp_path / "dst")
        eng = create_db_engine(src_db)
        with Session(eng) as s:
            _entity(s, "e1", "Roof")
            s.commit()

        snapshot = tmp_path / "snap.json"
        runner.invoke(app, ["export", "--format", "json", "--output", str(snapshot), "--db", src_db])
        json_text = snapshot.read_text()

        result = runner.invoke(app, ["import", "--input", "-", "--db", dst_db], input=json_text)
        assert result.exit_code == 0
        assert "entities: 1 added" in result.output

    def test_import_missing_file_errors(self, tmp_path):
        db = _init_db(tmp_path)
        result = runner.invoke(app, ["import", "--input", str(tmp_path / "nope.json"), "--db", db])
        assert result.exit_code != 0

    def test_import_bad_json_errors(self, tmp_path):
        db = _init_db(tmp_path)
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        result = runner.invoke(app, ["import", "--input", str(bad), "--db", db])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Cache CLI smoke tests
# ---------------------------------------------------------------------------

class TestCacheCLI:
    def test_cache_stats_empty(self, tmp_path):
        db = _init_db(tmp_path)
        result = runner.invoke(app, ["cache", "stats", "--db", db])
        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_cache_stats_json_empty(self, tmp_path):
        db = _init_db(tmp_path)
        result = runner.invoke(app, ["cache", "stats", "--json", "--db", db])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["entries"] == 0

    def test_cache_list_empty(self, tmp_path):
        db = _init_db(tmp_path)
        result = runner.invoke(app, ["cache", "list", "--db", db])
        assert result.exit_code == 0
        assert "No matching" in result.output

    def test_cache_clear_no_filters_errors(self, tmp_path):
        db = _init_db(tmp_path)
        result = runner.invoke(app, ["cache", "clear", "--db", db])
        assert result.exit_code != 0

    def test_cache_clear_no_matches(self, tmp_path):
        db = _init_db(tmp_path)
        result = runner.invoke(app, ["cache", "clear", "--model", "nonexistent", "--yes", "--db", db])
        assert result.exit_code == 0
        assert "No matching" in result.output
