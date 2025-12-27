"""Tests for consolidation strategies."""

import pydantic
import pytest
from typing import Any

from sieves.tasks.predictive.consolidation import (
    MultiEntityConsolidation,
    SingleEntityConsolidation,
    LabelScoreConsolidation,
    TextConsolidation,
    QAConsolidation,
    MapScoreConsolidation,
)


class DummyEntity(pydantic.BaseModel):
    """Dummy entity for testing."""
    name: str
    score: float | None


def test_multi_entity_consolidation():
    strategy = MultiEntityConsolidation(extractor=lambda res: res["entities"])
    results = [
        {"entities": [DummyEntity(name="A", score=0.8), DummyEntity(name="B", score=0.6)]},
        {"entities": [DummyEntity(name="A", score=0.4)]},
    ]
    offsets = [(0, 2)]

    consolidated = strategy.consolidate(results, offsets)
    assert len(consolidated) == 1
    entities = sorted(consolidated[0], key=lambda x: x.name)
    assert len(entities) == 2
    assert entities[0].name == "A"
    assert entities[0].score == pytest.approx(0.6)
    assert entities[1].name == "B"
    assert entities[1].score == 0.6


def test_single_entity_consolidation():
    strategy = SingleEntityConsolidation(extractor=lambda res: res["entity"])
    results = [
        {"entity": DummyEntity(name="A", score=0.8)},
        {"entity": DummyEntity(name="A", score=0.6)},
        {"entity": DummyEntity(name="B", score=0.9)},
    ]
    offsets = [(0, 3)]

    consolidated = strategy.consolidate(results, offsets)
    assert len(consolidated) == 1
    winner = consolidated[0]
    assert winner is not None
    assert winner.name == "A"
    assert winner.score == pytest.approx(0.7)


def test_label_score_consolidation():
    labels = ["positive", "negative", "neutral"]
    strategy = LabelScoreConsolidation(
        labels=labels,
        mode="multi",
        extractor=lambda res: res
    )

    results = [
        {"positive": 0.8, "negative": 0.1, "neutral": 0.1},
        {"positive": 0.6, "negative": 0.2, "neutral": 0.2},
    ]
    offsets = [(0, 2)]

    consolidated = strategy.consolidate(results, offsets)
    assert len(consolidated) == 1
    scores = consolidated[0]
    # positive: (0.8 + 0.6) / 2 = 0.7
    # negative: (0.1 + 0.2) / 2 = 0.15
    # neutral: (0.1 + 0.2) / 2 = 0.15
    assert scores[0] == ("positive", pytest.approx(0.7))
    # negative and neutral might be in any order due to same score, but positive must be first
    assert scores[0][0] == "positive"

    # Test single mode
    strategy_single = LabelScoreConsolidation(
        labels=labels,
        mode="single",
        extractor=lambda res: {res["label"]: res["score"]}
    )
    results_single = [
        {"label": "positive", "score": 0.8},
        {"label": "negative", "score": 0.2},
    ]
    consolidated_single = strategy_single.consolidate(results_single, [(0, 2)])
    assert len(consolidated_single) == 1
    # positive: 0.8 / 2 = 0.4
    # negative: 0.2 / 2 = 0.1
    # neutral: 0.0
    assert consolidated_single[0][0] == ("positive", pytest.approx(0.4))


def test_text_consolidation():
    strategy = TextConsolidation(extractor=lambda res: (res["text"], res.get("score")), joiner=" ")
    results = [
        {"text": "Hello", "score": 0.8},
        {"text": "world", "score": 0.4},
    ]
    offsets = [(0, 2)]

    consolidated = strategy.consolidate(results, offsets)
    assert len(consolidated) == 1
    text, score = consolidated[0]
    assert text == "Hello world"
    assert score == pytest.approx(0.6)


def test_qa_consolidation():
    questions = ["What?", "Who?"]
    strategy = QAConsolidation(
        questions=questions,
        extractor=lambda res: [(qa["q"], qa["a"], qa.get("s")) for qa in res["qa"]]
    )

    results = [
        {"qa": [{"q": "What?", "a": "A", "s": 0.8}, {"q": "Who?", "a": "B", "s": 0.6}]},
        {"qa": [{"q": "What?", "a": "is", "s": 0.2}, {"q": "Who?", "a": "C", "s": None}]},
    ]
    offsets = [(0, 2)]

    consolidated = strategy.consolidate(results, offsets)
    assert len(consolidated) == 1
    qa_list = consolidated[0]
    assert len(qa_list) == 2

    # What? -> "A is", score (0.8+0.2)/2 = 0.5
    assert qa_list[0][0] == "What?"
    assert qa_list[0][1] == "A is"
    assert qa_list[0][2] == pytest.approx(0.5)

    # Who? -> "B C", score 0.6 / 1 = 0.6
    assert qa_list[1][0] == "Who?"
    assert qa_list[1][1] == "B C"
    assert qa_list[1][2] == pytest.approx(0.6)


def test_map_score_consolidation():
    aspects = ["quality", "price"]
    strategy = MapScoreConsolidation(
        keys=aspects,
        extractor=lambda res: (res["aspects"], res.get("score"))
    )

    results = [
        {"aspects": {"quality": 0.8, "price": 0.4}, "score": 0.9},
        {"aspects": {"quality": 0.6, "price": 0.2}, "score": 0.7},
    ]
    offsets = [(0, 2)]

    consolidated = strategy.consolidate(results, offsets)
    assert len(consolidated) == 1
    aspect_scores, overall_score = consolidated[0]

    assert aspect_scores["quality"] == pytest.approx(0.7)
    assert aspect_scores["price"] == pytest.approx(0.3)
    assert overall_score == pytest.approx(0.8)
