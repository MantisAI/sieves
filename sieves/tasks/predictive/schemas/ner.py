"""Schemas for NER task."""

from __future__ import annotations

import dspy
import gliner2.inference.engine
import pydantic

from sieves.model_wrappers import dspy_, gliner_, langchain_, outlines_
from sieves.tasks.predictive.schemas.core import FewshotExample as BaseFewshotExample


class EntityWithContext(pydantic.BaseModel):
    """Entity mention with text span and type.

    Attributes:
        text: Entity text.
        context: Context around entity.
        entity_type: Type of entity.
        score: Confidence score.
    """

    text: str
    context: str
    entity_type: str
    score: float | None = None


class Entity(pydantic.BaseModel):
    """Class for storing entity information.

    Attributes:
        text: Entity text.
        start: Start offset.
        end: End offset.
        entity_type: Type of entity.
        score: Confidence score.
    """

    text: str
    start: int
    end: int
    entity_type: str
    score: float | None = None

    def __eq__(self, other: object) -> bool:
        """Compare two entities.

        :param other: Other entity to compare with.
        :return: True if entities are equal, False otherwise.
        """
        if not isinstance(other, Entity):
            return False
        return (
            self.start == other.start
            and self.end == other.end
            and self.text == other.text
            and self.entity_type == other.entity_type
        )

    def __hash__(self) -> int:
        """Compute entity hash.

        :returns: Entity hash.
        """
        return hash((self.start, self.end, self.text, self.entity_type))


class Entities(pydantic.BaseModel):
    """Collection of entities with associated text.

    Attributes:
        entities: List of entities.
        text: Source text.
    """

    entities: list[Entity]
    text: str


# --8<-- [start:Result]
class Result(Entities):
    """Result of an NER task."""

    pass


# --8<-- [end:Result]


class FewshotExample(BaseFewshotExample):
    """Few‑shot example with entities annotated in text.

    Attributes:
        text: Input text.
        entities: List of entities with context.
    """

    text: str
    entities: list[EntityWithContext]

    @property
    def target_fields(self) -> tuple[str, ...]:
        """Return target fields.

        :return: Target fields.
        """
        return ("entities",)


TaskModel = dspy_.Model | gliner_.Model | langchain_.Model | outlines_.Model
TaskPromptSignature = (
    type[dspy.Signature]
    | type[pydantic.BaseModel]
    | gliner2.inference.engine.Schema
    | gliner2.inference.engine.StructureBuilder
)
TaskResult = Result
