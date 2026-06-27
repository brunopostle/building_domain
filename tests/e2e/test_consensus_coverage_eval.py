"""Section 16.6 — Consensus coverage test.

Exercises the real ``bsos curate verify`` command (PROPOSAL.md §16.6, §16.1).
For each ground-truth assertion (``source_model='human'``) the command attempts
an exact ``(subject, predicate, object)`` key match against the extracted corpus
and falls back to embedding cosine similarity at a configurable threshold
(default 0.90). It reports coverage % and exits non-zero when coverage falls
below the target (default 0.80).

The project corpus carries no hand-curated ground truth yet, so these tests seed
a self-contained scratch DB rather than the real ``bsos.db``: a set of human
ground-truth assertions plus a corpus that key-matches a controlled fraction of
them. This pins the §16.6 mechanism — exact-match counting, the coverage target
gate, and the fuzzy-match fallback — without depending on a populated corpus.

The fuzzy fallback uses sentence-transformers; the tests skip cleanly when the
embedding model cannot be loaded (e.g. offline with an empty cache).
"""
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner
from sqlmodel import Session

from bsos.cli.main import app
from bsos.persistence.database import create_db_engine
from bsos.persistence.models import AssertionRow, EntityRow

pytest.importorskip("sentence_transformers")

runner = CliRunner(mix_stderr=False)
NOW = datetime.now(timezone.utc)

# Ten canonical ground-truth subject→predicate→object triples.
GT_TRIPLES = [
    ("roof", "requires", "structural-support"),
    ("window", "requires", "lintel"),
    ("staircase", "requires", "handrail"),
    ("foundation", "supports", "wall"),
    ("beam", "supports", "floor"),
    ("door", "connects_to", "corridor"),
    ("wall", "contains", "insulation"),
    ("drainage", "requires", "fall"),
    ("balcony", "requires", "guarding"),
    ("chimney", "requires", "flue"),
]


@pytest.fixture(scope="module")
def _embedding_model():
    """Skip the whole module if the default embedding model cannot be loaded."""
    from sentence_transformers import SentenceTransformer
    try:
        return SentenceTransformer("all-mpnet-base-v2")
    except Exception as exc:  # offline + uncached, or download failure
        pytest.skip(f"Embedding model unavailable for §16.6 fuzzy fallback: {exc}")


def _node_ids(triples):
    names = set()
    for s, _, o in triples:
        names.add(s)
        names.add(o)
    return names


def _init_db(tmp_path):
    db = tmp_path / "consensus_eval.db"
    runner.invoke(app, ["init", "--db", str(db), "--no-gitignore"])
    return str(db)


def _entity(eid, name):
    return EntityRow(
        id=eid, name=name, entity_type="concept",
        status="accepted", source_model="eval_harness", created_at=NOW,
    )


def _assertion(aid, s, p, o, source_model, status="accepted"):
    return AssertionRow(
        id=aid, subject_id=s, predicate=p, object_id=o,
        subject_type="concept", object_type="concept",
        status=status, confidence=0.95, knowledge_origin="architectural",
        source_model=source_model, created_at=NOW,
    )


def _seed(db_path, matched_count):
    """Seed all GT triples as human ground truth and key-match the first
    ``matched_count`` of them with corpus assertions."""
    eng = create_db_engine(db_path)
    with Session(eng) as s:
        for name in _node_ids(GT_TRIPLES):
            s.add(_entity(f"e-{name}", name))
        for i, (subj, pred, obj) in enumerate(GT_TRIPLES):
            s.add(_assertion(
                f"gt-{i}", f"e-{subj}", pred, f"e-{obj}", source_model="human"
            ))
            if i < matched_count:
                s.add(_assertion(
                    f"corpus-{i}", f"e-{subj}", pred, f"e-{obj}",
                    source_model="eval_corpus",
                ))
        s.commit()


def _verify(db, **opts):
    args = ["curate", "verify", "--db", db]
    for k, v in opts.items():
        args += [f"--{k}", str(v)]
    return runner.invoke(app, args)


def test_no_ground_truth_reports_nothing(tmp_path):
    """A corpus with no human rows reports nothing and exits cleanly."""
    db = _init_db(tmp_path)
    eng = create_db_engine(db)
    with Session(eng) as s:
        s.add(_entity("e-roof", "roof"))
        s.add(_entity("e-x", "structural-support"))
        s.add(_assertion("c0", "e-roof", "requires", "e-x", source_model="eval_corpus"))
        s.commit()

    result = _verify(db)
    assert result.exit_code == 0, result.output
    assert "No ground-truth assertions found" in result.output


def test_coverage_meets_target(tmp_path, _embedding_model):
    """8/10 exact matches == 80% target → PASS, exit 0."""
    db = _init_db(tmp_path)
    _seed(db, matched_count=8)

    result = _verify(db, threshold=0.90, target=0.80)
    assert result.exit_code == 0, result.output
    assert "Coverage: 8/10 (80%)" in result.output
    assert "PASS" in result.output


def test_coverage_below_target_warns(tmp_path, _embedding_model):
    """7/10 exact matches == 70% < 80% target → WARNING, exit 1."""
    db = _init_db(tmp_path)
    _seed(db, matched_count=7)

    result = _verify(db, threshold=0.90, target=0.80)
    assert result.exit_code == 1, result.output
    assert "Coverage: 7/10 (70%)" in result.output
    # WARNING is printed to stderr.
    assert "below target" in (result.stderr + result.output)


def test_fuzzy_fallback_recovers_near_misses(tmp_path, _embedding_model):
    """The embedding fallback matches GT that has no exact key match.

    Seed corpus assertions that carry the *same entity names* as the ground truth
    but under different entity ids: the ``(subject_id, predicate, object_id)``
    keys differ (no exact match), yet the embedded text is identical, so cosine
    similarity ≈ 1.0 clears the 0.90 threshold via the fuzzy branch.
    """
    db = _init_db(tmp_path)
    eng = create_db_engine(db)
    with Session(eng) as s:
        for name in _node_ids(GT_TRIPLES):
            s.add(_entity(f"e-{name}", name))          # GT-side ids
            s.add(_entity(f"c-{name}", name))          # corpus-side ids, same name
        for i, (subj, pred, obj) in enumerate(GT_TRIPLES):
            s.add(_assertion(f"gt-{i}", f"e-{subj}", pred, f"e-{obj}", source_model="human"))
            s.add(_assertion(f"corpus-{i}", f"c-{subj}", pred, f"c-{obj}", source_model="eval_corpus"))
        s.commit()

    result = _verify(db, threshold=0.90, target=0.80)
    # No exact key matches; every match comes from the fuzzy fallback (text ≈ 1.0).
    assert "Coverage: 10/10 (100%)" in result.output, result.output
    assert result.exit_code == 0, result.output
