"""Schemas for summarization task."""

from __future__ import annotations

import dspy
import pydantic

from sieves.model_wrappers import dspy_, langchain_, outlines_
from sieves.tasks.predictive.schemas.core import FewshotExample as BaseFewshotExample


class FewshotExample(BaseFewshotExample):
    """Example for summarization few-shot prompting.

    Attributes:
        text: Input text.
        summary: Summary of text.
    """

    summary: str

    @property
    def target_fields(self) -> tuple[str, ...]:
        """Return target fields.

        :return: Target fields.
        """
        return ("summary",)


# --8<-- [start:Result]
class Result(pydantic.BaseModel):
    """Result of a summarization task.

    Attributes:
        summary: Summary of text.
    """

    summary: str


# --8<-- [end:Result]


_TaskModel = dspy_.Model | langchain_.Model | outlines_.Model
_TaskPromptSignature = type[dspy.Signature] | type[pydantic.BaseModel]
_TaskResult = Result
