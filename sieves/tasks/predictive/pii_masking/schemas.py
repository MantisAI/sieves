"""Schemas for PII masking task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import override

import dspy
import pydantic

from sieves.model_wrappers import dspy_, langchain_, outlines_
from sieves.tasks.predictive.schemas import FewshotExample as BaseFewshotExample


class PIIEntity(pydantic.BaseModel, frozen=True):
    """PII entity."""

    entity_type: str
    text: str


class FewshotExample(BaseFewshotExample):
    """Example for PII masking few-shot prompting."""

    masked_text: str
    pii_entities: list[PIIEntity]

    @override
    @property
    def target_fields(self) -> Sequence[str]:
        return "masked_text", "pii_entities"


class Result(pydantic.BaseModel):
    """Result of a PII masking task."""

    masked_text: str
    pii_entities: list[PIIEntity]


_TaskModel = dspy_.Model | langchain_.Model | outlines_.Model
_TaskPromptSignature = pydantic.BaseModel | dspy.Signature
_TaskResult = Result
