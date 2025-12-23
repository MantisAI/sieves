"""Schemas for classification task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import override

import dspy
import gliner2.inference.engine
import pydantic

from sieves.model_wrappers import dspy_, gliner_, huggingface_, langchain_, outlines_
from sieves.tasks.predictive.schemas import FewshotExample as BaseFewshotExample


class FewshotExampleMultiLabel(BaseFewshotExample):
    """Few‑shot example for multi‑label classification with per‑label confidences."""

    confidence_per_label: dict[str, float]

    @override
    @property
    def target_fields(self) -> Sequence[str]:
        return ("confidence_per_label",)

    @pydantic.model_validator(mode="after")
    def check_confidence(self) -> FewshotExampleMultiLabel:
        """Validate that confidences lie within [0, 1]."""
        if any([conf for conf in self.confidence_per_label.values() if not 0 <= conf <= 1]):
            raise ValueError("Confidence has to be between 0 and 1.")
        return self


class FewshotExampleSingleLabel(BaseFewshotExample):
    """Few‑shot example for single‑label classification with a global confidence."""

    label: str
    confidence: float

    @override
    @property
    def target_fields(self) -> Sequence[str]:
        return ("label", "confidence")

    @pydantic.model_validator(mode="after")
    def check_confidence(self) -> FewshotExampleSingleLabel:
        """Check confidence value.

        Return:
            FewshotExampleSingleLabel instance.

        """
        if not (0 <= self.confidence <= 1):
            raise ValueError("Confidence has to be between 0 and 1.")
        return self


# --8<-- [start:Result]
class ResultSingleLabel(pydantic.BaseModel):
    """Result of a single-label classification task."""

    label: str
    score: float


class ResultMultiLabel(pydantic.BaseModel):
    """Result of a multi-label classification task."""

    label_scores: list[tuple[str, float]]


# --8<-- [end:Result]


_TaskModel = dspy_.Model | gliner_.Model | langchain_.Model | huggingface_.Model | outlines_.Model
_TaskPromptSignature = gliner2.inference.engine.Schema | pydantic.BaseModel | dspy.Signature
_TaskResult = ResultSingleLabel | ResultMultiLabel
