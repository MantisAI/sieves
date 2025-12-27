"""Schemas for sentiment analysis task."""

from __future__ import annotations

import dspy
import pydantic

from sieves.model_wrappers import dspy_, langchain_, outlines_
from sieves.tasks.predictive.schemas.core import FewshotExample as BaseFewshotExample


class FewshotExample(BaseFewshotExample):
    """Example for sentiment analysis few-shot prompting.

    Attributes:
        text: Input text.
        sentiment_per_aspect: Mapping of aspects to sentiments.
        score: Confidence score for the sentiment assessment.
    """

    sentiment_per_aspect: dict[str, float]
    score: float | None = None

    @property
    def target_fields(self) -> tuple[str, ...]:
        """Return target fields.

        :return: Target fields.
        """
        return ("sentiment_per_aspect", "score")


# --8<-- [start:Result]
class Result(pydantic.BaseModel):
    """Result of an aspect-based sentiment analysis task.

    Attributes:
        sentiment_per_aspect: Mapping of aspects to sentiment scores.
        score: Overall confidence score for the sentiment analysis.
    """

    sentiment_per_aspect: dict[str, float] = pydantic.Field(
        description="A mapping where keys are aspect names (e.g., 'overall', 'quality') and values are sentiment "
        "scores between 0 (Negative) and 1 (Positive)."
    )
    score: float | None = pydantic.Field(
        default=None, description="Provide an overall confidence score for the sentiment analysis, between 0 and 1."
    )


# --8<-- [end:Result]


TaskModel = dspy_.Model | langchain_.Model | outlines_.Model
TaskPromptSignature = type[dspy.Signature] | type[pydantic.BaseModel]
TaskResult = Result
