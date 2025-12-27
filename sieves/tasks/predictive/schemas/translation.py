"""Schemas for translation task."""

from __future__ import annotations

import dspy
import pydantic

from sieves.model_wrappers import dspy_, langchain_, outlines_
from sieves.tasks.predictive.schemas.core import FewshotExample as BaseFewshotExample


class FewshotExample(BaseFewshotExample):
    """Example for translation few-shot prompting.

    Attributes:
        text: Input text.
        translation: Translated text.
        score: Confidence score.
    """

    translation: str
    score: float | None = None

    @property
    def target_fields(self) -> tuple[str, ...]:
        """Return target fields.

        :return: Target fields.
        """
        return ("translation",)


# --8<-- [start:Result]
class Result(pydantic.BaseModel):
    """Result of a translation task. Contains the translated text and a confidence score.

    Attributes:
        translation: Translated text.
        score: Confidence score.
    """

    translation: str = pydantic.Field(description="The input text translated into the target language.")
    score: float | None = pydantic.Field(
        default=None, description="Provide a confidence score for the translation, between 0 and 1."
    )


# --8<-- [end:Result]


TaskModel = dspy_.Model | langchain_.Model | outlines_.Model
TaskPromptSignature = type[dspy.Signature] | type[pydantic.BaseModel]
TaskResult = Result
