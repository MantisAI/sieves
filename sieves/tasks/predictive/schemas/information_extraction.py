"""Schemas for information extraction task."""

from __future__ import annotations

import dspy
import gliner2.inference.engine
import pydantic

from sieves.model_wrappers import dspy_, gliner_, langchain_, outlines_
from sieves.tasks.predictive.schemas.core import FewshotExample as BaseFewshotExample


class FewshotExampleMulti(BaseFewshotExample):
    """Few-shot example for multi-entity extraction.

    Attributes:
        text: Input text.
        entities: List of entities.
    """

    entities: list[pydantic.BaseModel]

    @property
    def target_fields(self) -> tuple[str, ...]:
        """Return target fields.

        :return: Target fields.
        """
        return ("entities",)


class FewshotExampleSingle(BaseFewshotExample):
    """Few-shot example for single-entity extraction.

    Attributes:
        text: Input text.
        entity: Extracted entity.
    """

    entity: pydantic.BaseModel | None

    @property
    def target_fields(self) -> tuple[str, ...]:
        """Return target fields.

        :return: Target fields.
        """
        return ("entity",)


# --8<-- [start:Result]
class ResultSingle(pydantic.BaseModel):
    """Result of a single-entity extraction task.

    Attributes:
        entity: Extracted entity.
    """

    entity: pydantic.BaseModel | None


class ResultMulti(pydantic.BaseModel):
    """Result of a multi-entity extraction task.

    Attributes:
        entities: List of extracted entities.
    """

    entities: list[pydantic.BaseModel]


# --8<-- [end:Result]


TaskModel = dspy_.Model | gliner_.Model | langchain_.Model | outlines_.Model
TaskPromptSignature = (
    type[dspy.Signature]
    | type[pydantic.BaseModel]
    | gliner2.inference.engine.Schema
    | gliner2.inference.engine.StructureBuilder
)
TaskResult = ResultSingle | ResultMulti
