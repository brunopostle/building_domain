"""Unit tests for vocabulary, predicate registry, and threshold constants."""
import pytest
from bsos.vocab import (
    CORE_PREDICATES, PREDICATE_REGISTRY, SPATIAL_RELATION_TYPES, PredicateDefinition,
)
from bsos.pipeline.pass2 import CLUSTER_DISTANCE_THRESHOLD
from bsos.pipeline.pass3 import CONSISTENCY_THRESHOLD


# ---------------------------------------------------------------------------
# CORE_PREDICATES
# ---------------------------------------------------------------------------

def test_core_predicates_is_frozenset():
    assert isinstance(CORE_PREDICATES, frozenset)


def test_core_predicates_contains_expected():
    for pred in ("requires", "depends_on", "protects_from", "unsuitable_for",
                 "improves", "conflicts_with", "contains", "connects_to", "supports"):
        assert pred in CORE_PREDICATES


def test_core_predicates_count():
    assert len(CORE_PREDICATES) == 9


# ---------------------------------------------------------------------------
# PREDICATE_REGISTRY
# ---------------------------------------------------------------------------

def test_registry_has_all_core_predicates():
    for pred in CORE_PREDICATES:
        assert pred in PREDICATE_REGISTRY, f"Missing from registry: {pred}"


def test_registry_values_are_predicate_definitions():
    for key, val in PREDICATE_REGISTRY.items():
        assert isinstance(val, PredicateDefinition)


def test_registry_predicate_matches_key():
    for key, val in PREDICATE_REGISTRY.items():
        assert val.predicate == key


def test_directional_predicates():
    assert PREDICATE_REGISTRY["requires"].directional is True
    assert PREDICATE_REGISTRY["depends_on"].directional is True
    assert PREDICATE_REGISTRY["protects_from"].directional is True
    assert PREDICATE_REGISTRY["supports"].directional is True


def test_symmetric_predicates():
    assert PREDICATE_REGISTRY["conflicts_with"].directional is False
    assert PREDICATE_REGISTRY["connects_to"].directional is False


def test_registry_meaning_non_empty():
    for key, val in PREDICATE_REGISTRY.items():
        assert len(val.meaning) > 10, f"Meaning too short for {key}"


# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------

def test_cluster_distance_threshold_range():
    """Clustering threshold must be a valid cosine distance (0, 2)."""
    assert 0.0 < CLUSTER_DISTANCE_THRESHOLD < 2.0


def test_cluster_distance_threshold_value():
    assert CLUSTER_DISTANCE_THRESHOLD == pytest.approx(0.20)


def test_consistency_threshold_range():
    """Consistency threshold must be a valid cosine similarity (0, 1)."""
    assert 0.0 < CONSISTENCY_THRESHOLD < 1.0


def test_consistency_threshold_value():
    assert CONSISTENCY_THRESHOLD == pytest.approx(0.70)


def test_thresholds_compatible():
    """Clustering distance < 1 - consistency threshold would be contradictory."""
    # cos_dist = 1 - cos_sim; two assertions matched at cos_sim >= 0.70
    # means cos_dist <= 0.30; entity dedup at cos_dist <= 0.20 is tighter
    assert CLUSTER_DISTANCE_THRESHOLD < (1.0 - CONSISTENCY_THRESHOLD + 0.20)


# ---------------------------------------------------------------------------
# SPATIAL_RELATION_TYPES
# ---------------------------------------------------------------------------

def test_spatial_relation_types_is_list():
    assert isinstance(SPATIAL_RELATION_TYPES, list)


def test_spatial_relation_types_non_empty():
    assert len(SPATIAL_RELATION_TYPES) > 0


def test_spatial_contains_overlap_with_core():
    """Some spatial relations overlap with core predicates (contains, connects_to)."""
    overlap = set(SPATIAL_RELATION_TYPES) & CORE_PREDICATES
    assert len(overlap) > 0
