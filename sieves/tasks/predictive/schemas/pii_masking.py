"""Schemas for PII masking task."""

from __future__ import annotations

import dspy
import pydantic

from sieves.model_wrappers import dspy_, langchain_, outlines_
from sieves.tasks.predictive.schemas.core import FewshotExample as BaseFewshotExample


class PIIEntity(pydantic.BaseModel, frozen=True):
    """Personally Identifiable Information (PII) entity.

    Attributes:
        entity_type: Type of PII.
        text: Entity text.
        score: Confidence score.
    """

    entity_type: str = pydantic.Field(description="The type of PII identified (e.g., EMAIL, PHONE, SSN).")
    text: str = pydantic.Field(description="The original text of the PII entity.")
    score: float | None = pydantic.Field(
        default=None, description="Provide a confidence score for the PII identification, between 0 and 1."
    )


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
    """Result of a PII masking task. Contains the masked text and the identified PII entities.

    PII entities should be masked with [MASKED].

    Attributes:
        masked_text: Masked version of text.
        pii_entities: List of PII entities.
    """

    masked_text: str = pydantic.Field(description="The original text with PII entities replaced by placeholders.")
    pii_entities: list[PIIEntity] = pydantic.Field(description="List of all PII entities identified in the text.")


# --8<-- [end:Result]


TaskModel = dspy_.Model | langchain_.Model | outlines_.Model
TaskPromptSignature = type[dspy.Signature] | type[pydantic.BaseModel]
TaskResult = Result
