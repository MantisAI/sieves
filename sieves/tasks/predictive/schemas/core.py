"""Base schemas and type variables for predictive tasks."""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Self, TypeVar

import dspy
import pydantic

from sieves.model_wrappers import ModelWrapper, ModelWrapperInferenceMode

TaskPromptSignature = TypeVar("TaskPromptSignature", covariant=True)
TaskResult = TypeVar("TaskResult")
TaskBridge = TypeVar("TaskBridge", bound="Bridge[TaskPromptSignature, TaskResult, ModelWrapperInferenceMode]")  # type: ignore[valid-type]  # noqa: F821


class EvaluationSignature(dspy.Signature):
    """Evaluate similarity between ground truth and predicted outputs."""

    target_fields: str = dspy.InputField(desc="Names of output fields being compared.")
    ground_truth: str = dspy.InputField(desc="Ground truth output values.")
    prediction: str = dspy.InputField(desc="Predicted output values.")

    similarity_score: float = dspy.OutputField(
        desc="Similarity score between 0.0 and 1.0, where 1.0 means identical and 0.0 means completely different."
    )


class FewshotExample(pydantic.BaseModel):
    """Few-shot example.

    Attributes:
        text: Input text.
    """

    text: str

    @property
    def input_fields(self) -> Sequence[str]:
        """Defines which fields are inputs.

        :return: Sequence of field names.
        """
        return ("text",)

    @property
    @abc.abstractmethod
    def target_fields(self) -> Sequence[str]:
        """Define which fields are targets, i.e. the end results the task aims to produce.

        :return: Sequence of field names.
        """

    def to_dspy(self) -> dspy.Example:
        """Convert to `dspy.Example`.

        :returns: Example as `dspy.Example`.
        """
        return dspy.Example(**ModelWrapper.convert_fewshot_examples([self])[0]).with_inputs(*self.input_fields)

    @classmethod
    def from_dspy(cls, example: dspy.Example) -> Self:
        """Convert from `dspy.Example`.

        :param example: Example as `dspy.Example`.
        :returns: Example as `FewshotExample`.
        """
        return cls(**example)
