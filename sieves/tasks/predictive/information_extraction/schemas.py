"""Schemas for information extraction task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import override

import dspy
import gliner2.inference.engine
import pydantic

from sieves.model_wrappers import dspy_, gliner_, langchain_, outlines_
from sieves.tasks.predictive.schemas import FewshotExample as BaseFewshotExample


class FewshotExampleMulti(BaseFewshotExample):
    """Few-shot example for multi-entity extraction."""

    entities: list[pydantic.BaseModel]

    @override
    @property
    def target_fields(self) -> Sequence[str]:
        return ("entities",)


class FewshotExampleSingle(BaseFewshotExample):
    """Few-shot example for single-entity extraction."""

    entity: pydantic.BaseModel | None

    @override
    @property
    def target_fields(self) -> Sequence[str]:
        return ("entity",)


# --8<-- [start:Result]
class ResultSingle(pydantic.BaseModel):
    """Result of a single-entity extraction task."""

    entity: pydantic.BaseModel | None


class ResultMulti(pydantic.BaseModel):
    """Result of a multi-entity extraction task."""

    entities: list[pydantic.BaseModel]


# --8<-- [end:Result]


_TaskModel = dspy_.Model | gliner_.Model | langchain_.Model | outlines_.Model
_TaskPromptSignature = pydantic.BaseModel | dspy.Signature | gliner2.inference.engine.Schema
_TaskResult = ResultSingle | ResultMulti
