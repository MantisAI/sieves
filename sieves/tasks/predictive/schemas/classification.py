"""Schemas for classification task."""

from __future__ import annotations

import dspy
import gliner2.inference.engine
import pydantic

from sieves.model_wrappers import dspy_, gliner_, huggingface_, langchain_, outlines_
from sieves.tasks.predictive.schemas.core import FewshotExample as BaseFewshotExample


class FewshotExampleMultiLabel(BaseFewshotExample):
    """Few‑shot example for multi‑label classification with per‑label confidences.

    Attributes:
        text: Input text.
        confidence_per_label: Mapping of labels to confidence scores.
    """

    confidence_per_label: dict[str, float]

    @pydantic.model_validator(mode="after")
    def check_confidence(self) -> FewshotExampleMultiLabel:
        """Validate that confidences lie within [0, 1].

        :return: Validated instance.
        """
        if any([conf for conf in self.confidence_per_label.values() if not 0 <= conf <= 1]):
            raise ValueError("Confidence has to be between 0 and 1.")
        return self

    @property
    def target_fields(self) -> tuple[str, ...]:
        """Return target fields.

        :return: Target fields.
        """
        return ("confidence_per_label",)


class FewshotExampleSingleLabel(BaseFewshotExample):
    """Few‑shot example for single‑label classification with a global confidence.

    Attributes:
        text: Input text.
        label: Predicted label.
        confidence: Confidence score.
    """

    label: str
    confidence: float

    @pydantic.model_validator(mode="after")
    def check_confidence(self) -> FewshotExampleSingleLabel:
        """Check confidence value.

        :return: Validated instance.
        """
        if not (0 <= self.confidence <= 1):
            raise ValueError("Confidence has to be between 0 and 1.")
        return self

    @property
    def target_fields(self) -> tuple[str, ...]:
        """Return target fields.

        :return: Target fields.
        """
        return ("label", "confidence")


# --8<-- [start:Result]
class ResultSingleLabel(pydantic.BaseModel):
    """Result of a single-label classification task.

    Attributes:
        label: Predicted label.
        score: Confidence score.
    """

    label: str
    score: float


class ResultMultiLabel(pydantic.BaseModel):
    """Result of a multi-label classification task.

    Attributes:
        label_scores: List of label-score pairs.
    """

    label_scores: list[tuple[str, float]]


# --8<-- [end:Result]


_TaskModel = dspy_.Model | gliner_.Model | langchain_.Model | huggingface_.Model | outlines_.Model
_TaskPromptSignature = (
    type[dspy.Signature]
    | type[pydantic.BaseModel]
    | gliner2.inference.engine.Schema
    | gliner2.inference.engine.StructureBuilder
    | list[str]
)
_TaskResult = ResultSingleLabel | ResultMultiLabel
