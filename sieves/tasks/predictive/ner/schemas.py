"""Schemas for NER task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, override

import pydantic

from sieves.model_wrappers import dspy_, gliner_, langchain_, outlines_
from sieves.tasks.predictive.schemas import FewshotExample as BaseFewshotExample


class EntityWithContext(pydantic.BaseModel):
    """Entity mention with text span and type."""

    text: str
    context: str
    entity_type: str


class Entity(pydantic.BaseModel):
    """Class for storing entity information."""

    text: str
    start: int
    end: int
    entity_type: str

    def __eq__(self, other: object) -> bool:
        """Compare two entities.

        :param other: Other entity to compare with.
        :return: True if entities are equal, False otherwise.
        """
        if not isinstance(other, Entity):
            return False
        # Two entities are equal if they have the same start, end, text and entity_type
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


class Result(pydantic.BaseModel):
    """Collection of entities with associated text."""

    entities: list[Entity]
    text: str


class FewshotExample(BaseFewshotExample):
    """Few‑shot example with entities annotated in text."""

    text: str
    entities: list[EntityWithContext]

    @override
    @property
    def target_fields(self) -> Sequence[str]:
        return ("entities",)


_TaskModel = dspy_.Model | gliner_.Model | langchain_.Model | outlines_.Model
_TaskPromptSignature = Any
_TaskResult = Result
