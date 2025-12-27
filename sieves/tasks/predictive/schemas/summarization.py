"""Schemas for summarization task."""

from __future__ import annotations

from collections.abc import Sequence

import dspy
import pydantic

from sieves.model_wrappers import dspy_, langchain_, outlines_
from sieves.tasks.predictive.schemas.core import FewshotExample as BaseFewshotExample


class FewshotExample(BaseFewshotExample):
    """Example for summarization few-shot prompting.

    Attributes:
        text: Input text.
        n_words: Guideline for summary length.
        summary: Summary of text.
        score: Confidence score.
    """

    n_words: int
    summary: str
    score: float | None = None

    @property
    def input_fields(self) -> Sequence[str]:
        """Return input fields.

        :return: Input fields.
        """
        return "text", "n_words"

    @property
    def target_fields(self) -> tuple[str, ...]:
        """Return target fields.

        :return: Target fields.
        """
        return ("summary",)


# --8<-- [start:Result]
class Result(pydantic.BaseModel):
    """Result of a summarization task. Contains the generated summary and a confidence score.

    Attributes:
        summary: Summary of text.
        score: Confidence score.
    """

    summary: str = pydantic.Field(description="The generated summary of the input text.")
    score: float | None = pydantic.Field(
        default=None, description="Provide a confidence score for the generated summary, between 0 and 1."
    )


# --8<-- [end:Result]


TaskModel = dspy_.Model | langchain_.Model | outlines_.Model
TaskPromptSignature = type[dspy.Signature] | type[pydantic.BaseModel]
TaskResult = Result
