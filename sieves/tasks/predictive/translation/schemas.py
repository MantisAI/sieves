"""Schemas for translation task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import override

import dspy
import pydantic

from sieves.model_wrappers import dspy_, langchain_, outlines_
from sieves.tasks.predictive.schemas import FewshotExample as BaseFewshotExample


class FewshotExample(BaseFewshotExample):
    """Few-shot example with a target translation."""

    to: str
    translation: str

    @override
    @property
    def input_fields(self) -> Sequence[str]:
        return "text", "to"

    @override
    @property
    def target_fields(self) -> Sequence[str]:
        return ("translation",)


# --8<-- [start:Result]
class Result(pydantic.BaseModel):
    """Result of a translation task."""

    translation: str


# --8<-- [end:Result]


_TaskModel = dspy_.Model | langchain_.Model | outlines_.Model
_TaskPromptSignature = pydantic.BaseModel | dspy.Signature
_TaskResult = Result
