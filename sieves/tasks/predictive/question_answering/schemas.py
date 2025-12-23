"""Schemas for question answering task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import override

import dspy
import pydantic

from sieves.model_wrappers import dspy_, langchain_, outlines_
from sieves.tasks.predictive.schemas import FewshotExample as BaseFewshotExample


class FewshotExample(BaseFewshotExample):
    """Few-shot example with questions and answers for a context."""

    questions: tuple[str, ...] | list[str]
    answers: tuple[str, ...] | list[str]

    @override
    @property
    def input_fields(self) -> Sequence[str]:
        return "text", "questions"

    @override
    @property
    def target_fields(self) -> Sequence[str]:
        return ("answers",)


class Result(pydantic.BaseModel):
    """Result of a question-answering task."""

    answers: list[str]


_TaskModel = dspy_.Model | langchain_.Model | outlines_.Model
_TaskPromptSignature = pydantic.BaseModel | dspy.Signature
_TaskResult = Result
