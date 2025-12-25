"""Schemas for relation extraction task."""

from __future__ import annotations

import dspy
import gliner2.inference.engine
import pydantic

from sieves.model_wrappers import dspy_, gliner_, langchain_, outlines_
from sieves.tasks.predictive.schemas.core import FewshotExample as BaseFewshotExample


class RelationEntity(pydantic.BaseModel, frozen=True):
    """Entity involved in a relation.

    Attributes:
        text: Surface text of the entity.
        entity_type: Type of the entity.
    """

    text: str
    entity_type: str


class RelationTriplet(pydantic.BaseModel, frozen=True):
    """Triplet representing a relation between two entities.

    Attributes:
        head: The subject entity.
        relation: The type of relation.
        tail: The object entity.
        score: Confidence score.
    """

    head: RelationEntity
    relation: str
    tail: RelationEntity
    score: float | None = None


# --8<-- [start:Result]
class Result(pydantic.BaseModel):
    """Result of a relation extraction task.

    Attributes:
        triplets: List of extracted relation triplets.
    """

    triplets: list[RelationTriplet]


# --8<-- [end:Result]


class RelationEntityWithContext(pydantic.BaseModel):
    """Entity mention with text span, type, and context for span discovery.

    Attributes:
        text: Surface text of the entity.
        context: Short context around the entity.
        entity_type: Type of the entity.
    """

    text: str
    context: str
    entity_type: str


class RelationTripletWithContext(pydantic.BaseModel):
    """Triplet with context for span discovery.

    Attributes:
        head: The head entity with context.
        relation: The relation type.
        tail: The tail entity with context.
        score: Confidence score.
    """

    head: RelationEntityWithContext
    relation: str
    tail: RelationEntityWithContext
    score: float | None = None


class FewshotExample(BaseFewshotExample):
    """Few-shot example for relation extraction.

    Attributes:
        text: Input text.
        triplets: Expected relation triplets.
    """

    triplets: list[RelationTriplet]

    @property
    def target_fields(self) -> tuple[str, ...]:
        """Return target fields.

        :return: Target fields.
        """
        return ("triplets",)


TaskModel = dspy_.Model | gliner_.Model | langchain_.Model | outlines_.Model
TaskPromptSignature = (
    type[dspy.Signature]
    | type[pydantic.BaseModel]
    | gliner2.inference.engine.Schema
    | gliner2.inference.engine.StructureBuilder
)
TaskResult = Result
