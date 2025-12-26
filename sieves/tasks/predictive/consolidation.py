"""Consolidation strategies for predictive tasks."""

from __future__ import annotations

import abc
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, TypeVar

import pydantic

_EntityType = TypeVar("_EntityType", bound=pydantic.BaseModel)


class ConsolidationStrategy(abc.ABC):
    """Abstract base class for consolidation strategies."""

    @abc.abstractmethod
    def consolidate(self, results: Sequence[Any], docs_offsets: list[tuple[int, int]]) -> Sequence[Any]:
        """Consolidate chunk results into document results.

        :param results: Sequence of raw chunk results.
        :param docs_offsets: List of (start, end) offsets mapping chunks to documents.
        :return: Sequence of consolidated "clean" results.
        """


class MultiEntityConsolidation(ConsolidationStrategy):
    """Consolidation strategy for multiple entities."""

    def __init__(self, extractor: Callable[[Any], Iterable[pydantic.BaseModel]]):
        """Initialize MultiEntityConsolidation.

        :param extractor: Callable to extract a list of entities from a raw chunk result.
        """
        self.extractor = extractor

    def consolidate(
        self, results: Sequence[Any], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[list[pydantic.BaseModel]]:
        """Consolidate multiple entities from chunks.

        :param results: Raw chunk results.
        :param docs_offsets: Chunk offsets per document.
        :return: Consolidated list of entities per document.
        """
        consolidated_results: list[list[pydantic.BaseModel]] = []
        for start, end in docs_offsets:
            entities: list[pydantic.BaseModel] = []
            for res in results[start:end]:
                if res is None:
                    continue
                entities.extend(self.extractor(res))

            consolidated_results.append(self._consolidate_entities(entities))
        return consolidated_results

    @staticmethod
    def _consolidate_entities(entities: list[_EntityType]) -> list[_EntityType]:
        """Deduplicate and average scores for entities.

        Logic moved from sieves.tasks.predictive.utils.consolidate_entities_multi.
        """
        if not entities:
            return []

        entities_map: dict[str, tuple[_EntityType, list[float]]] = {}

        for entity in entities:
            if entity is None:
                continue

            assert isinstance(entity, pydantic.BaseModel)
            key_entity = entity.model_copy(update={"score": None})
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


class SingleEntityConsolidation(ConsolidationStrategy):
    """Consolidation strategy for a single entity."""

    def __init__(self, extractor: Callable[[Any], pydantic.BaseModel | None]):
        """Initialize SingleEntityConsolidation.

        :param extractor: Callable to extract a single entity from a raw chunk result.
        """
        self.extractor = extractor

    def consolidate(
        self, results: Sequence[Any], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[pydantic.BaseModel | None]:
        """Consolidate single entities from chunks via majority vote.

        :param results: Raw chunk results.
        :param docs_offsets: Chunk offsets per document.
        :return: Winner entity per document.
        """
        consolidated_results: list[pydantic.BaseModel | None] = []
        for start, end in docs_offsets:
            entities_with_indices: list[tuple[pydantic.BaseModel | None, int]] = []
            for i, res in enumerate(results[start:end]):
                if res is None:
                    entities_with_indices.append((None, i))
                    continue
                entities_with_indices.append((self.extractor(res), i))

            consolidated_results.append(self._consolidate_single(entities_with_indices))
        return consolidated_results

    @staticmethod
    def _consolidate_single(entities_with_indices: list[tuple[_EntityType | None, int]]) -> _EntityType | None:
        """Majority vote for single entity.

        Logic moved from sieves.tasks.predictive.utils.consolidate_entities_single.
        """
        if not entities_with_indices:
            return None

        entity_counts: Counter[str] = Counter()
        key_to_entity: dict[str, _EntityType | None] = {}
        first_seen: dict[str, int] = {}
        scores_per_entity: dict[str, list[float]] = {}

        for entity, i in entities_with_indices:
            if entity is not None:
                key_entity = entity.model_copy(update={"score": None})
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

        max_count = max(entity_counts.values())
        candidates = [k for k, count in entity_counts.items() if count == max_count]
        winner_key = min(candidates, key=lambda k: first_seen[k])

        winner_entity = key_to_entity[winner_key]
        if winner_entity is None:
            return None

        scores = scores_per_entity[winner_key]
        avg_score = sum(scores) / len(scores) if scores else None

        return winner_entity.model_copy(update={"score": avg_score})


class LabelScoreConsolidation(ConsolidationStrategy):
    """Consolidation strategy for classification tasks."""

    def __init__(
        self,
        labels: list[str],
        mode: Literal["single", "multi"],
        extractor: Callable[[Any], dict[str, float]],
    ):
        """Initialize LabelScoreConsolidation.

        :param labels: List of valid labels.
        :param mode: Classification mode ('single' or 'multi').
        :param extractor: Callable to extract label scores from a raw chunk result.
        """
        self.labels = labels
        self.mode = mode
        self.extractor = extractor

    def consolidate(
        self, results: Sequence[Any], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[list[tuple[str, float]]]:
        """Consolidate label scores from chunks.

        :param results: Raw chunk results.
        :param docs_offsets: Chunk offsets per document.
        :return: Sorted list of (label, score) pairs per document.
        """
        consolidated_results: list[list[tuple[str, float]]] = []
        for start, end in docs_offsets:
            label_scores: dict[str, float] = {label: 0.0 for label in self.labels}
            chunk_results = results[start:end]
            num_chunks = end - start

            for res in chunk_results:
                if res is None:
                    continue

                scores = self.extractor(res)
                for label, score in scores.items():
                    if label in label_scores:
                        # Clamp score to [0, 1] as in existing bridges.
                        label_scores[label] += max(0.0, min(float(score), 1.0))

            # Average score, sort by it in descending order.
            avg_scores = [(label, score / num_chunks) for label, score in label_scores.items()]
            # Sort by score descending.
            sorted_scores = sorted(avg_scores, key=lambda x: x[1], reverse=True)
            consolidated_results.append(sorted_scores)

        return consolidated_results


class TextConsolidation(ConsolidationStrategy):
    """Consolidation strategy for tasks producing text (translation, summarization)."""

    def __init__(self, extractor: Callable[[Any], tuple[str, float | None]], joiner: str = "\n"):
        """Initialize TextConsolidation.

        :param extractor: Callable to extract (text, score) from a raw chunk result.
        :param joiner: String used to join text chunks.
        """
        self.extractor = extractor
        self.joiner = joiner

    def consolidate(
        self, results: Sequence[Any], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[tuple[str, float | None]]:
        """Consolidate text chunks.

        :param results: Raw chunk results.
        :param docs_offsets: Chunk offsets per document.
        :return: Consolidated (text, average_score) per document.
        """
        consolidated_results: list[tuple[str, float | None]] = []
        for start, end in docs_offsets:
            texts: list[str] = []
            scores: list[float] = []

            for res in results[start:end]:
                if res is None:
                    continue
                text, score = self.extractor(res)
                texts.append(text)
                if score is not None:
                    scores.append(score)

            avg_score = sum(scores) / len(scores) if scores else None
            consolidated_results.append((self.joiner.join(texts).strip(), avg_score))

        return consolidated_results


class QAConsolidation(ConsolidationStrategy):
    """Consolidation strategy for question answering."""

    def __init__(self, questions: list[str], extractor: Callable[[Any], Iterable[tuple[str, str, float | None]]]):
        """Initialize QAConsolidation.

        :param questions: List of questions.
        :param extractor: Callable to extract (question, answer, score) tuples from a raw chunk result.
        """
        self.questions = questions
        self.extractor = extractor

    def consolidate(
        self, results: Sequence[Any], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[list[tuple[str, str, float | None]]]:
        """Consolidate QA pairs.

        :param results: Raw chunk results.
        :param docs_offsets: Chunk offsets per document.
        :return: Consolidated list of (question, answer, score) per document.
        """
        consolidated_results: list[list[tuple[str, str, float | None]]] = []
        for start, end in docs_offsets:
            qa_map: dict[str, tuple[list[str], list[float]]] = {q: ([], []) for q in self.questions}

            for res in results[start:end]:
                if res is None:
                    continue
                for q, a, s in self.extractor(res):
                    if q in qa_map:
                        qa_map[q][0].append(a)
                        if s is not None:
                            qa_map[q][1].append(s)

            consolidated_qa: list[tuple[str, str, float | None]] = []
            for question in self.questions:
                answers, scores = qa_map[question]
                avg_score = sum(scores) / len(scores) if scores else None
                consolidated_qa.append((question, " ".join(answers).strip(), avg_score))

            consolidated_results.append(consolidated_qa)

        return consolidated_results


class MapScoreConsolidation(ConsolidationStrategy):
    """Consolidation strategy for map-based scores (e.g. sentiment analysis)."""

    def __init__(
        self,
        keys: Iterable[str],
        extractor: Callable[[Any], tuple[dict[str, float], float | None]],
    ):
        """Initialize MapScoreConsolidation.

        :param keys: Keys (aspects/labels) to average.
        :param extractor: Callable to extract (map_scores, overall_score) from a raw chunk result.
        """
        self.keys = list(keys)
        self.extractor = extractor

    def consolidate(
        self, results: Sequence[Any], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[tuple[dict[str, float], float | None]]:
        """Consolidate map-based scores.

        :param results: Raw chunk results.
        :param docs_offsets: Chunk offsets per document.
        :return: Consolidated (avg_map_scores, avg_overall_score) per document.
        """
        consolidated_results: list[tuple[dict[str, float], float | None]] = []
        for start, end in docs_offsets:
            key_scores: dict[str, float] = {k: 0.0 for k in self.keys}
            overall_scores: list[float] = []
            num_chunks = end - start

            for res in results[start:end]:
                if res is None:
                    continue
                m_scores, o_score = self.extractor(res)
                for k, s in m_scores.items():
                    if k in key_scores:
                        key_scores[k] += max(0.0, min(float(s), 1.0))
                if o_score is not None:
                    overall_scores.append(o_score)

            avg_key_scores = {k: s / num_chunks for k, s in key_scores.items()}
            avg_overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else None
            consolidated_results.append((avg_key_scores, avg_overall_score))

        return consolidated_results
