"""Tests for predictive utilities and consolidation strategies."""

import pydantic
import pytest

from sieves.tasks.predictive.consolidation import (
    MultiEntityConsolidation,
    SingleEntityConsolidation,
)


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


def test_multi_entity_consolidation_empty():
    strategy = MultiEntityConsolidation(extractor=lambda x: x)
    # 0 chunks for 1 document
    assert strategy.consolidate([], [(0, 0)]) == [[]]


def test_multi_entity_consolidation_single():
    entity = DummyEntity(name="test", label="LABEL", score=0.8)
    strategy = MultiEntityConsolidation(extractor=lambda x: [x])
    results = strategy.consolidate([entity], [(0, 1)])
    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0].name == "test"
    assert results[0][0].score == 0.8


def test_multi_entity_consolidation_deduplication_and_averaging():
    entities = [
        DummyEntity(name="test", label="LABEL", score=0.8),
        DummyEntity(name="test", label="LABEL", score=0.6),
        DummyEntity(name="other", label="LABEL", score=0.9),
    ]
    strategy = MultiEntityConsolidation(extractor=lambda x: [x])
    results = strategy.consolidate(entities, [(0, 3)])
    assert len(results) == 1
    doc_results = results[0]
    assert len(doc_results) == 2

    # Sort results by name for deterministic testing
    doc_results = sorted(doc_results, key=lambda x: x.name)

    assert doc_results[0].name == "other"
    assert doc_results[0].score == pytest.approx(0.9)

    assert doc_results[1].name == "test"
    assert doc_results[1].score == pytest.approx(0.7)


def test_multi_entity_consolidation_mixed_scores():
    entities = [
        DummyEntity(name="test", label="LABEL", score=0.8),
        DummyEntity(name="test", label="LABEL", score=None),
    ]
    strategy = MultiEntityConsolidation(extractor=lambda x: [x])
    results = strategy.consolidate(entities, [(0, 2)])
    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0].score == pytest.approx(0.8)  # Only one score was provided


def test_multi_entity_consolidation_with_none():
    entities = [
        DummyEntity(name="test", label="LABEL", score=0.8),
        None,
    ]
    strategy = MultiEntityConsolidation(extractor=lambda x: [x] if x else [])
    results = strategy.consolidate(entities, [(0, 2)])
    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0].name == "test"


def test_multi_entity_consolidation_non_frozen():
    entities = [
        NonFrozenEntity(name="test", score=0.8),
        NonFrozenEntity(name="test", score=0.4),
    ]
    strategy = MultiEntityConsolidation(extractor=lambda x: [x])
    results = strategy.consolidate(entities, [(0, 2)])
    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0].name == "test"
    assert results[0][0].score == pytest.approx(0.6)


def test_multi_entity_consolidation_nested():
    dummy1 = DummyEntity(name="inner", label="L")
    entities = [
        NestedEntity(name="outer", details=dummy1, score=0.8),
        NestedEntity(name="outer", details=dummy1, score=0.2),
    ]
    strategy = MultiEntityConsolidation(extractor=lambda x: [x])
    results = strategy.consolidate(entities, [(0, 2)])
    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0].score == pytest.approx(0.5)


def test_single_entity_consolidation_empty():
    strategy = SingleEntityConsolidation(extractor=lambda x: x)
    assert strategy.consolidate([], [(0, 0)]) == [None]


def test_single_entity_consolidation_majority():
    entities = [
        DummyEntity(name="A", label="L", score=0.8),
        DummyEntity(name="A", label="L", score=0.6),
        DummyEntity(name="B", label="L", score=0.9),
    ]
    strategy = SingleEntityConsolidation(extractor=lambda x: x)
    results = strategy.consolidate(entities, [(0, 3)])
    winner = results[0]
    assert winner is not None
    assert winner.name == "A"
    assert winner.score == pytest.approx(0.7)


def test_single_entity_consolidation_tie_break():
    # Tie between A and B, A seen first at index 0
    entities = [
        DummyEntity(name="A", label="L", score=0.8),
        DummyEntity(name="B", label="L", score=0.9),
    ]
    strategy = SingleEntityConsolidation(extractor=lambda x: x)
    results = strategy.consolidate(entities, [(0, 2)])
    winner = results[0]
    assert winner is not None
    assert winner.name == "A"


def test_single_entity_consolidation_none_handling():
    entities = [
        DummyEntity(name="A", label="L", score=0.8),
        None,
        None,
    ]
    strategy = SingleEntityConsolidation(extractor=lambda x: x)
    results = strategy.consolidate(entities, [(0, 3)])
    winner = results[0]
    assert winner is None  # None won by majority (2 vs 1)


def test_single_entity_consolidation_all_none():
    entities = [
        None,
        None,
    ]
    strategy = SingleEntityConsolidation(extractor=lambda x: x)
    results = strategy.consolidate(entities, [(0, 2)])
    winner = results[0]
    assert winner is None
