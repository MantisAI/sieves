"""Tests for predictive utilities."""

import pydantic
import pytest

from sieves.tasks.predictive.utils import consolidate_entities_multi, consolidate_entities_single


class DummyEntity(pydantic.BaseModel):
    """Dummy entity for testing."""

    name: str
    label: str
    score: float | None = None


class NonFrozenEntity(pydantic.BaseModel):
    """Non-frozen entity for testing."""

    name: str
    score: float | None = None


class NestedEntity(pydantic.BaseModel):
    """Nested entity for testing."""

    name: str
    details: DummyEntity
    score: float | None = None


def test_consolidate_entities_multi_empty():
    assert consolidate_entities_multi([]) == []


def test_consolidate_entities_multi_single():
    entity = DummyEntity(name="test", label="LABEL", score=0.8)
    results = consolidate_entities_multi([entity])
    assert len(results) == 1
    assert results[0].name == "test"
    assert results[0].score == 0.8


def test_consolidate_entities_multi_deduplication_and_averaging():
    entities = [
        DummyEntity(name="test", label="LABEL", score=0.8),
        DummyEntity(name="test", label="LABEL", score=0.6),
        DummyEntity(name="other", label="LABEL", score=0.9),
    ]
    results = consolidate_entities_multi(entities)
    assert len(results) == 2

    # Sort results by name for deterministic testing
    results = sorted(results, key=lambda x: x.name)

    assert results[0].name == "other"
    assert results[0].score == pytest.approx(0.9)

    assert results[1].name == "test"
    assert results[1].score == pytest.approx(0.7)


def test_consolidate_entities_multi_mixed_scores():
    entities = [
        DummyEntity(name="test", label="LABEL", score=0.8),
        DummyEntity(name="test", label="LABEL", score=None),
    ]
    results = consolidate_entities_multi(entities)
    assert len(results) == 1
    assert results[0].score == pytest.approx(0.8)  # Only one score was provided


def test_consolidate_entities_multi_with_none():
    entities = [
        DummyEntity(name="test", label="LABEL", score=0.8),
        None,
    ]
    # We ignore None in multi mode.
    results = consolidate_entities_multi(entities)  # type: ignore[arg-type]
    assert len(results) == 1
    assert results[0].name == "test"


def test_consolidate_entities_multi_non_frozen():
    entities = [
        NonFrozenEntity(name="test", score=0.8),
        NonFrozenEntity(name="test", score=0.4),
    ]
    results = consolidate_entities_multi(entities)
    assert len(results) == 1
    assert results[0].name == "test"
    assert results[0].score == pytest.approx(0.6)


def test_consolidate_entities_multi_nested():
    dummy1 = DummyEntity(name="inner", label="L")
    entities = [
        NestedEntity(name="outer", details=dummy1, score=0.8),
        NestedEntity(name="outer", details=dummy1, score=0.2),
    ]
    results = consolidate_entities_multi(entities)
    assert len(results) == 1
    assert results[0].score == pytest.approx(0.5)


def test_consolidate_entities_single_empty():
    assert consolidate_entities_single([]) is None


def test_consolidate_entities_single_majority():
    entities_with_indices = [
        (DummyEntity(name="A", label="L", score=0.8), 0),
        (DummyEntity(name="A", label="L", score=0.6), 1),
        (DummyEntity(name="B", label="L", score=0.9), 2),
    ]
    winner = consolidate_entities_single(entities_with_indices)
    assert winner is not None
    assert winner.name == "A"
    assert winner.score == pytest.approx(0.7)


def test_consolidate_entities_single_tie_break():
    # Tie between A and B, A seen first at index 0
    entities_with_indices = [
        (DummyEntity(name="A", label="L", score=0.8), 0),
        (DummyEntity(name="B", label="L", score=0.9), 1),
    ]
    winner = consolidate_entities_single(entities_with_indices)
    assert winner is not None
    assert winner.name == "A"


def test_consolidate_entities_single_none_handling():
    entities_with_indices = [
        (DummyEntity(name="A", label="L", score=0.8), 0),
        (None, 1),
        (None, 2),
    ]
    winner = consolidate_entities_single(entities_with_indices)
    assert winner is None  # None won by majority (2 vs 1)


def test_consolidate_entities_single_all_none():
    entities_with_indices = [
        (None, 0),
        (None, 1),
    ]
    winner = consolidate_entities_single(entities_with_indices)
    assert winner is None
