"""Schemas for question answering task."""

from __future__ import annotations

import dspy
import pydantic

from sieves.model_wrappers import dspy_, langchain_, outlines_
from sieves.tasks.predictive.schemas.core import FewshotExample as BaseFewshotExample


class FewshotExample(BaseFewshotExample):
    """Few-shot example with questions and answers for a context.

    Attributes:
        text: Input text.
        questions: Questions asked.
        answers: Expected answers.
    """

    questions: tuple[str, ...] | list[str]
    answers: tuple[str, ...] | list[str]

    @property
    def input_fields(self) -> tuple[str, ...]:
        """Return input fields.

        :return: Input fields.
        """
        return ("text", "questions")

    @property
    def target_fields(self) -> tuple[str, ...]:
        """Return target fields.

        :return: Target fields.
        """
        return ("answers",)


# --8<-- [start:Result]
class Result(pydantic.BaseModel):
    """Result of a question-answering task.

    Attributes:
        answers: List of answers.
    """

    answers: list[str]


# --8<-- [end:Result]


_TaskModel = dspy_.Model | langchain_.Model | outlines_.Model
_TaskPromptSignature = type[dspy.Signature] | type[pydantic.BaseModel]
_TaskResult = Result
