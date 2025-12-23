"""Schemas for sentiment analysis task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import override

import dspy
import pydantic

from sieves.model_wrappers import dspy_, langchain_, outlines_
from sieves.tasks.predictive.schemas import FewshotExample as BaseFewshotExample


class FewshotExample(BaseFewshotExample):
    """Few-shot example with per-aspect sentiment scores."""

    sentiment_per_aspect: dict[str, float]

    @override
    @property
    def target_fields(self) -> Sequence[str]:
        return ("sentiment_per_aspect",)

    @pydantic.model_validator(mode="after")
    def check_confidence(self) -> FewshotExample:
        """Validate that 'overall' exists and all scores are in [0, 1]."""
        assert "overall" in self.sentiment_per_aspect, ValueError(
            "'overall' score has to be given in `sentiment_per_aspect` dict."
        )
        if any([conf for conf in self.sentiment_per_aspect.values() if not 0 <= conf <= 1]):
            raise ValueError("Sentiment score has to be between 0 and 1.")
        return self


class Result(pydantic.BaseModel):
    """Result of a sentiment analysis task."""

    sentiment_per_aspect: dict[str, float]


_TaskModel = dspy_.Model | langchain_.Model | outlines_.Model
_TaskPromptSignature = pydantic.BaseModel | dspy.Signature
_TaskResult = Result
