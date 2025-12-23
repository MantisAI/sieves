"""Schemas for PII masking task."""

from __future__ import annotations

import dspy
import pydantic

from sieves.model_wrappers import dspy_, langchain_, outlines_
from sieves.tasks.predictive.schemas.core import FewshotExample as BaseFewshotExample


class PIIEntity(pydantic.BaseModel, frozen=True):
    """PII entity.

    Attributes:
        entity_type: Type of PII.
        text: Entity text.
    """

    entity_type: str
    text: str


class FewshotExample(BaseFewshotExample):
    """Example for PII masking few-shot prompting.

    Attributes:
        text: Input text.
        masked_text: Masked version of text.
        pii_entities: List of PII entities.
    """

    masked_text: str
    pii_entities: list[PIIEntity]

    @property
    def target_fields(self) -> tuple[str, ...]:
        """Return target fields.

        :return: Target fields.
        """
        return ("masked_text", "pii_entities")


# --8<-- [start:Result]
class Result(pydantic.BaseModel):
    """Result of a PII masking task.

    Attributes:
        masked_text: Masked version of text.
        pii_entities: List of PII entities.
    """

    masked_text: str
    pii_entities: list[PIIEntity]


# --8<-- [end:Result]


_TaskModel = dspy_.Model | langchain_.Model | outlines_.Model
_TaskPromptSignature = type[dspy.Signature] | type[pydantic.BaseModel]
_TaskResult = Result
