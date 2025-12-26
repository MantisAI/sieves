"""Utilities for predictive tasks."""

from __future__ import annotations

import warnings
from typing import TypeVar

import pydantic

from sieves.tasks.predictive.consolidation import (
    MultiEntityConsolidation,
    SingleEntityConsolidation,
)

_EntityType = TypeVar("_EntityType", bound=pydantic.BaseModel)


def consolidate_entities_multi(entities: list[_EntityType]) -> list[_EntityType]:
    """Consolidate entities found in multiple chunks by deduplicating and averaging scores.

    .. deprecated:: 1.0.0
        Use :class:`sieves.tasks.predictive.consolidation.MultiEntityConsolidation` instead.

    :param entities: List of entities found across all chunks of a document.
    :return: List of deduplicated entities with averaged scores.
    """
    warnings.warn(
        "consolidate_entities_multi is deprecated and will be removed in a future version. "
        "Use MultiEntityConsolidation strategy instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return MultiEntityConsolidation._consolidate_entities(entities)


def consolidate_entities_single(entities_with_indices: list[tuple[_EntityType | None, int]]) -> _EntityType | None:
    """Consolidate single entity results from multiple chunks by majority vote and averaging scores.

    .. deprecated:: 1.0.0
        Use :class:`sieves.tasks.predictive.consolidation.SingleEntityConsolidation` instead.

    :param entities_with_indices: List of (entity, index) pairs found across all chunks of a document.
    :return: Winner entity with averaged scores, or None.
    """
    warnings.warn(
        "consolidate_entities_single is deprecated and will be removed in a future version. "
        "Use SingleEntityConsolidation strategy instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return SingleEntityConsolidation._consolidate_single(entities_with_indices)
