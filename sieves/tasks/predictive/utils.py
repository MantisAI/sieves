"""Utilities for predictive tasks."""

from __future__ import annotations

from collections import Counter
from typing import TypeVar

import pydantic

_EntityType = TypeVar("_EntityType", bound=pydantic.BaseModel)


def consolidate_entities_multi(entities: list[_EntityType]) -> list[_EntityType]:
    """Consolidate entities found in multiple chunks by deduplicating and averaging scores.

    :param entities: List of entities found across all chunks of a document.
    :return: List of deduplicated entities with averaged scores.
    """
    if not entities:
        return []

    # entities_map: entity_json -> (entity_without_score, list of scores)
    entities_map: dict[str, tuple[_EntityType, list[float]]] = {}

    for entity in entities:
        if entity is None:
            continue

        # Create a copy without score for deduplication key.
        assert isinstance(entity, pydantic.BaseModel)
        key_entity = entity.model_copy(update={"score": None})

        # Use JSON representation as hashable key.
        key = key_entity.model_dump_json() if hasattr(key_entity, "model_dump_json") else str(key_entity)

        if key not in entities_map:
            entities_map[key] = (key_entity, [])
        if getattr(entity, "score", None) is not None:
            entities_map[key][1].append(entity.score)

    consolidated_entities: list[_EntityType] = []
    for key_entity, scores in entities_map.values():
        avg_score = sum(scores) / len(scores) if scores else None
        if hasattr(key_entity, "model_copy"):
            consolidated_entities.append(key_entity.model_copy(update={"score": avg_score}))
        else:
            consolidated_entities.append(key_entity)

    return consolidated_entities


def consolidate_entities_single(entities_with_indices: list[tuple[_EntityType | None, int]]) -> _EntityType | None:
    """Consolidate single entity results from multiple chunks by majority vote and averaging scores.

    :param entities_with_indices: List of (entity, index) pairs found across all chunks of a document.
    :return: Winner entity with averaged scores, or None.
    """
    if not entities_with_indices:
        return None

    # entity_counts: entity_json -> count.
    entity_counts: Counter[str] = Counter()
    # key_to_entity: entity_json -> entity_without_score
    key_to_entity: dict[str, _EntityType | None] = {}
    first_seen: dict[str, int] = {}
    scores_per_entity: dict[str, list[float]] = {}

    for entity, i in entities_with_indices:
        # Deduplicate key by removing score.
        if entity is not None:
            key_entity = entity.model_copy(update={"score": None})
            # Use JSON representation as hashable key.
            key = key_entity.model_dump_json() if hasattr(key_entity, "model_dump_json") else str(key_entity)
        else:
            key_entity = None
            key = "null"

        entity_counts[key] += 1
        if key not in key_to_entity:
            key_to_entity[key] = key_entity
        if key not in first_seen:
            first_seen[key] = i
        if key not in scores_per_entity:
            scores_per_entity[key] = []
        if entity is not None and getattr(entity, "score", None) is not None:
            scores_per_entity[key].append(entity.score)

    if not entity_counts:
        return None

    # Pick winner by majority.
    max_count = max(entity_counts.values())
    candidates = [k for k, count in entity_counts.items() if count == max_count]
    winner_key = min(candidates, key=lambda k: first_seen[k])

    winner_entity = key_to_entity[winner_key]
    if winner_entity is None:
        return None

    scores = scores_per_entity[winner_key]
    avg_score = sum(scores) / len(scores) if scores else None
    return winner_entity.model_copy(update={"score": avg_score})
