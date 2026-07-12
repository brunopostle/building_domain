"""Unit tests for the report-specific helpers in
scripts/ifc_compliance_report.py.

The deterministic check engine (constraint classification, evaluation,
antipatterns) now lives in bsos.validation and is tested in test_validation.py.
What remains here is the report's own geometry helpers, MEP vocabulary, and
(building_domain-l5w.1) the semantic space-entity resolver that replaced the
old hardcoded SPACE_TO_ENTITY / SPATIAL_OBJECT_TO_USAGE lookup tables.
"""
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from sqlmodel import Session

from bsos.persistence.database import create_db_engine
from bsos.persistence.models import EntityRow, EmbeddingRow, ProcessRelationRow
from bsos.mcp_server.server import SEARCH_EMBEDDING_MODEL

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "ifc_compliance_report", ROOT / "scripts" / "ifc_compliance_report.py"
)
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)

NOW = datetime.now(timezone.utc)


# ── footprint area from triangulated mesh ────────────────────────────────────

import math


class _FakeGeom:
    """Minimal stand-in for an ifcopenshell triangulation (flat verts/faces)."""
    def __init__(self, verts, faces):
        self.verts = verts
        self.faces = faces


def _square_floor(side: float, angle: float = 0.0) -> _FakeGeom:
    """A horizontal square floor of `side`, rotated `angle` rad about Z,
    as two upward-facing (CCW-from-above) triangles."""
    pts = [(0, 0), (side, 0), (side, side), (0, side)]
    ca, sa = math.cos(angle), math.sin(angle)
    verts = []
    for x, y in pts:
        verts += [x * ca - y * sa, x * sa + y * ca, 0.0]
    return _FakeGeom(verts, [0, 1, 2, 0, 2, 3])


def test_footprint_area_of_axis_aligned_square():
    assert cr._footprint_area(_square_floor(3.0)) == pytest.approx(9.0)


def test_footprint_area_is_rotation_invariant():
    # A room whose plan is not aligned to X/Y must report its true area, not an
    # inflated bounding-box footprint.
    assert cr._footprint_area(_square_floor(3.0, math.radians(30))) == pytest.approx(9.0)


def test_footprint_area_ignores_downward_faces():
    # A floor face (CCW, upward) plus its reversed copy (downward) — only the
    # upward face counts, so the footprint is the floor area, not double.
    g = _square_floor(2.0)
    g.faces = list(g.faces) + [2, 1, 0, 3, 2, 0]  # reversed winding = downward
    assert cr._footprint_area(g) == pytest.approx(4.0)


# ── Semantic space-entity resolution (building_domain-l5w.1) ────────────────
# Replaces the old fixed SPACE_TO_ENTITY / SPATIAL_OBJECT_TO_USAGE dicts with
# embedding similarity, so a stub embedder (fixed vectors per query text) gives
# deterministic, fast tests without loading the real sentence-transformers model.

DIM = 4


def _vec(values):
    v = np.array(values, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v


class _StubEmbedder:
    """Fixed vector per query text; unlisted text falls back to a neutral vector."""
    _VECTORS = {
        "kitchen": [1.0, 0.0, 0.0, 0.0],
        "corridor": [0.0, 1.0, 0.0, 0.0],
        "nonsense": [0.0, 0.0, 0.0, 1.0],
    }

    def encode(self, texts):
        default = [0.25, 0.25, 0.25, 0.25]
        return np.array(
            [_vec(self._VECTORS.get(t.lower(), default)) for t in texts],
            dtype=np.float32,
        )


@pytest.fixture
def engine(tmp_path):
    return create_db_engine(str(tmp_path / "test.db"))


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s


def _add_space_entity(session, eid, name, vector):
    session.add(EntityRow(id=eid, name=name, entity_type="space",
                          source_model="test", created_at=NOW))
    session.add(EmbeddingRow(item_type="entity", item_id=eid, model=SEARCH_EMBEDDING_MODEL,
                             dim=DIM, content_hash="test", vector=_vec(vector).tobytes()))
    session.commit()


def test_semantic_match_entity_finds_best_match(session):
    _add_space_entity(session, "e-kitchen", "Kitchen", [1.0, 0.0, 0.0, 0.0])
    _add_space_entity(session, "e-corridor", "Corridor", [0.0, 1.0, 0.0, 0.0])
    assert cr.semantic_match_entity(session, "kitchen", entity_type="space",
                                    _embedder=_StubEmbedder()) == "Kitchen"


def test_semantic_match_entity_below_threshold_returns_none(session):
    _add_space_entity(session, "e-kitchen", "Kitchen", [1.0, 0.0, 0.0, 0.0])
    assert cr.semantic_match_entity(session, "nonsense", entity_type="space",
                                    min_score=0.9, _embedder=_StubEmbedder()) is None


def test_semantic_match_entity_filters_by_entity_type(session):
    session.add(EntityRow(id="e-component", name="Kitchen Counter",
                          entity_type="component", source_model="test", created_at=NOW))
    session.add(EmbeddingRow(item_type="entity", item_id="e-component", model=SEARCH_EMBEDDING_MODEL,
                             dim=DIM, content_hash="test", vector=_vec([1.0, 0.0, 0.0, 0.0]).tobytes()))
    session.commit()
    assert cr.semantic_match_entity(session, "kitchen", entity_type="space",
                                    _embedder=_StubEmbedder()) is None


def test_get_entity_type_returns_none_for_unknown_or_merged(tmp_path):
    import sqlite3
    db_path = tmp_path / "entity_type.db"
    engine = create_db_engine(str(db_path))
    with Session(engine) as s:
        s.add(EntityRow(id="e1", name="Kitchen", entity_type="space",
                        source_model="test", created_at=NOW))
        s.add(EntityRow(id="e2", name="Old Name", entity_type="space", status="merged",
                        source_model="test", created_at=NOW))
        s.commit()
    assert cr.get_entity_type(db_path, "Kitchen") == "space"
    assert cr.get_entity_type(db_path, "Old Name") is None
    assert cr.get_entity_type(db_path, "Nonexistent") is None


def test_mep_presence_keys_are_known_system_objects():
    # The report's ifc-side system map must use the shared vocabulary, or
    # evaluate() will silently treat a system as an unknown object.
    from bsos import validation
    assert set(cr.MEP_PRESENCE) <= validation.SYSTEM_OBJECTS


def test_space_local_systems_are_known_mep_systems():
    # Every space-local system must also have a building-wide mapping, so the
    # two presence paths stay in sync.
    assert set(cr.SYSTEM_SPACE_FIXTURES) <= set(cr.MEP_PRESENCE)


# ── Per-space MEP presence (building_domain-d91) ─────────────────────────────
# Minimal duck-typed stand-ins for ifcopenshell entities: build_facts and
# count_bounded_by_type only touch .BoundedBy, .RelatedBuildingElement, .is_a()
# and ifc.by_type().

class _FakeElem:
    def __init__(self, cls):
        self._cls = cls

    def is_a(self, other=None):
        return self._cls if other is None else self._cls == other


class _FakeRel:
    def __init__(self, elem):
        self.RelatedBuildingElement = elem


class _FakeSpace:
    def __init__(self, *bounding_classes):
        self.Name = "S"
        self.BoundedBy = [_FakeRel(_FakeElem(c)) for c in bounding_classes]

    def id(self):
        return 1


class _FakeIfc:
    """ifc.by_type(cls) returns model-wide elements of that class."""
    def __init__(self, **counts):
        self._by_type = {cls: [_FakeElem(cls)] * n for cls, n in counts.items()}

    def by_type(self, cls):
        return self._by_type.get(cls, [])


@pytest.fixture(autouse=True)
def _clear_mep_cache():
    cr._mep_cache.clear()
    yield
    cr._mep_cache.clear()


def test_drainage_requires_fixture_in_this_space():
    # A sanitary terminal exists in the model but does NOT bound this space:
    # the old building-wide check passed drainage everywhere — now it must not.
    ifc = _FakeIfc(IfcSanitaryTerminal=1)
    no_fixture = _FakeSpace("IfcWall", "IfcWindow")
    assert "Drainage System" not in cr.build_facts(ifc, no_fixture)["systems_present"]


def test_drainage_passes_when_fixture_bounds_the_space():
    ifc = _FakeIfc(IfcSanitaryTerminal=1)
    with_fixture = _FakeSpace("IfcWall", "IfcSanitaryTerminal")
    assert "Drainage System" in cr.build_facts(ifc, with_fixture)["systems_present"]


def test_distribution_systems_stay_building_wide():
    # Electrical/lighting are not space-local; a model-wide element counts even
    # if it does not bound the space.
    ifc = _FakeIfc(IfcLightFixture=1)
    space = _FakeSpace("IfcWall")
    systems = cr.build_facts(ifc, space)["systems_present"]
    assert "Lighting System" in systems
    assert "Electrical System" in systems


# ── Design-time prerequisite check (building_domain-l5w.4) ──────────────────

class _FakeProduct:
    def __init__(self, name=None, object_type=None):
        self.Name = name
        self.ObjectType = object_type


class _FakeMaterial:
    def __init__(self, name):
        self.Name = name


class _FakeIfcModel:
    """ifc.by_type(cls) returns model-wide products/materials by class."""
    def __init__(self, products=None, materials=None):
        self._products = products or []
        self._materials = materials or []

    def by_type(self, cls):
        if cls == "IfcProduct":
            return self._products
        if cls == "IfcMaterial":
            return self._materials
        return []


class _PrereqStubEmbedder:
    """Fixed vector per query text; unlisted text falls back to a neutral vector."""
    _VECTORS = {
        "waterproofing": [1.0, 0.0, 0.0, 0.0],
        "waterproof membrane": [1.0, 0.0, 0.0, 0.0],
        "roof covering": [0.0, 1.0, 0.0, 0.0],
        "interior finishes": [0.0, 0.0, 1.0, 0.0],
    }

    def encode(self, texts):
        default = [0.25, 0.25, 0.25, 0.25]
        return np.array(
            [_vec(self._VECTORS.get(t.lower(), default)) for t in texts],
            dtype=np.float32,
        )


def test_collect_model_evidence_texts_dedupes_case_insensitively():
    ifc = _FakeIfcModel(
        products=[
            _FakeProduct(name="Waterproof Membrane"),
            _FakeProduct(name="waterproof membrane"),
            _FakeProduct(object_type="Roof Covering"),
            _FakeProduct(name=None, object_type=None),
        ],
        materials=[_FakeMaterial("Bitumen")],
    )
    texts = [t.lower() for t in cr.collect_model_evidence_texts(ifc)]
    assert texts.count("waterproof membrane") == 1
    assert "roof covering" in texts
    assert "bitumen" in texts


def test_best_text_match_finds_match_above_threshold():
    texts = ["Roof Covering", "Waterproof Membrane"]
    vecs = _PrereqStubEmbedder().encode(texts)
    evidenced, matched, score = cr._best_text_match(
        "Waterproofing", texts, vecs, _PrereqStubEmbedder())
    assert evidenced is True
    assert matched == "Waterproof Membrane"
    assert score == pytest.approx(1.0)


def test_best_text_match_below_threshold_not_evidenced():
    texts = ["Roof Covering"]
    vecs = _PrereqStubEmbedder().encode(texts)
    evidenced, matched, score = cr._best_text_match(
        "Waterproofing", texts, vecs, _PrereqStubEmbedder(), min_score=0.9)
    assert evidenced is False


def test_best_text_match_empty_texts_returns_false():
    evidenced, matched, score = cr._best_text_match(
        "Waterproofing", [], np.empty((0, 4), dtype=np.float32), _PrereqStubEmbedder())
    assert (evidenced, matched, score) == (False, None, None)


def test_get_process_predecessors_dedupes_to_highest_confidence(tmp_path):
    db_path = tmp_path / "predecessors.db"
    engine = create_db_engine(str(db_path))
    with Session(engine) as s:
        s.add(EntityRow(id="e-waterproof", name="Waterproofing", entity_type="activity",
                        source_model="test", created_at=NOW))
        s.add(EntityRow(id="e-finishes", name="Interior Finishes", entity_type="activity",
                        source_model="test", created_at=NOW))
        s.add(ProcessRelationRow(id="pr1", predecessor_id="e-waterproof", successor_id="e-finishes",
                                 hard_constraint=True, source_model="modelA", created_at=NOW,
                                 confidence=0.6, status="accepted", knowledge_origin="physical",
                                 rationale="lower confidence edge"))
        s.add(ProcessRelationRow(id="pr2", predecessor_id="e-waterproof", successor_id="e-finishes",
                                 hard_constraint=True, source_model="modelB", created_at=NOW,
                                 confidence=0.9, status="accepted", knowledge_origin="physical",
                                 rationale="higher confidence edge"))
        s.commit()
    result = cr.get_process_predecessors(db_path, "Interior Finishes")
    assert result == [("Waterproofing", True, 0.9, "higher confidence edge")]


def test_get_process_predecessors_excludes_deprecated(tmp_path):
    db_path = tmp_path / "deprecated.db"
    engine = create_db_engine(str(db_path))
    with Session(engine) as s:
        s.add(EntityRow(id="e-waterproof", name="Waterproofing", entity_type="activity",
                        source_model="test", created_at=NOW))
        s.add(EntityRow(id="e-finishes", name="Interior Finishes", entity_type="activity",
                        source_model="test", created_at=NOW))
        s.add(ProcessRelationRow(id="pr1", predecessor_id="e-waterproof", successor_id="e-finishes",
                                 hard_constraint=True, source_model="test", created_at=NOW,
                                 confidence=0.9, status="deprecated", knowledge_origin="physical",
                                 rationale="stale edge"))
        s.commit()
    assert cr.get_process_predecessors(db_path, "Interior Finishes") == []


def test_check_prerequisites_report_entity_not_found(tmp_path):
    db_path = tmp_path / "empty.db"
    create_db_engine(str(db_path))
    result = cr.check_prerequisites_report(
        Path("dummy.ifc"), db_path, "Nonexistent", _embedder=_PrereqStubEmbedder())
    assert result == {"error": "entity_not_found", "query": "Nonexistent"}


def test_check_prerequisites_report_flags_missing_hard_prerequisite(tmp_path, monkeypatch):
    db_path = tmp_path / "prereq.db"
    engine = create_db_engine(str(db_path))
    with Session(engine) as s:
        s.add(EntityRow(id="e-waterproof", name="Waterproofing", entity_type="activity",
                        source_model="test", created_at=NOW))
        s.add(EntityRow(id="e-finishes", name="Interior Finishes", entity_type="activity",
                        source_model="test", created_at=NOW))
        s.add(ProcessRelationRow(id="pr1", predecessor_id="e-waterproof", successor_id="e-finishes",
                                 hard_constraint=True, source_model="test", created_at=NOW,
                                 confidence=0.9, status="accepted", knowledge_origin="physical",
                                 rationale="waterproofing must precede interior finishes"))
        s.commit()

    fake_ifc = _FakeIfcModel(products=[_FakeProduct(name="Roof Covering")])
    monkeypatch.setattr(cr.ifcopenshell, "open", lambda path: fake_ifc)

    result = cr.check_prerequisites_report(
        Path("dummy.ifc"), db_path, "interior finishes", _embedder=_PrereqStubEmbedder())

    assert result["entity"] == "Interior Finishes"
    assert result["prerequisites"] == [{
        "prerequisite": "Waterproofing",
        "hard_constraint": True,
        "confidence": 0.9,
        "rationale": "waterproofing must precede interior finishes",
        "evidenced": False,
        "matched_text": "Roof Covering",
        "match_score": 0.0,
    }]
    assert result["summary"] == {
        "total_prerequisites": 1, "evidenced": 0, "missing": 1, "missing_hard_constraints": 1,
    }
    assert result["recommendation"].startswith("Hold")


def test_check_prerequisites_report_evidenced_prerequisite_ok_to_proceed(tmp_path, monkeypatch):
    db_path = tmp_path / "prereq_ok.db"
    engine = create_db_engine(str(db_path))
    with Session(engine) as s:
        s.add(EntityRow(id="e-waterproof", name="Waterproofing", entity_type="activity",
                        source_model="test", created_at=NOW))
        s.add(EntityRow(id="e-finishes", name="Interior Finishes", entity_type="activity",
                        source_model="test", created_at=NOW))
        s.add(ProcessRelationRow(id="pr1", predecessor_id="e-waterproof", successor_id="e-finishes",
                                 hard_constraint=True, source_model="test", created_at=NOW,
                                 confidence=0.9, status="accepted", knowledge_origin="physical",
                                 rationale="waterproofing must precede interior finishes"))
        s.commit()

    fake_ifc = _FakeIfcModel(products=[_FakeProduct(name="Waterproof Membrane")])
    monkeypatch.setattr(cr.ifcopenshell, "open", lambda path: fake_ifc)

    result = cr.check_prerequisites_report(
        Path("dummy.ifc"), db_path, "Interior Finishes", _embedder=_PrereqStubEmbedder())

    assert result["prerequisites"][0]["evidenced"] is True
    assert result["summary"]["missing_hard_constraints"] == 0
    assert result["recommendation"] == "OK to proceed"
