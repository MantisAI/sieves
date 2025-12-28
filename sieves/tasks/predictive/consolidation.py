"""Consolidation strategies for predictive tasks."""

from __future__ import annotations

import abc
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, TypeVar

import pydantic

_EntityType = TypeVar("_EntityType", bound=pydantic.BaseModel)


def _average_scores(scores: list[float]) -> float | None:
    """Calculate the average of a list of scores.

    :param scores: List of float scores.
    :return: Average score or None if list is empty.
    """
    return sum(scores) / len(scores) if scores else None


def _get_entity_key(entity: pydantic.BaseModel) -> str:
    """Generate a hashable key for a Pydantic model by excluding the 'score' field.

    :param entity: The Pydantic model to key.
    :return: JSON string representation of the model without its score.
    """
    return entity.model_copy(update={"score": None}).model_dump_json()


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
        self._extractor = extractor

    def consolidate(
        self,
        results: Sequence[Any],
        docs_offsets: list[tuple[int, int]],
    ) -> Sequence[list[pydantic.BaseModel]]:
        """Consolidate multiple entities from chunks.

        :param results: Raw chunk results.
        :param docs_offsets: Chunk offsets per document.
        :return: Consolidated list of entities per document.
        """
        consolidated_results: list[list[pydantic.BaseModel]] = []
        for start, end in docs_offsets:
            entities = [e for res in results[start:end] if res is not None for e in self._extractor(res)]
            consolidated_results.append(self._consolidate_entities(entities))
        return consolidated_results

    @staticmethod
    def _consolidate_entities(entities: list[_EntityType]) -> list[_EntityType]:
        """Deduplicate entities and average their scores."""
        if not entities:
            return []

        entities_map: dict[str, tuple[_EntityType, list[float]]] = {}
        for entity in entities:
            if entity is None:
                continue

            key = _get_entity_key(entity)
            if key not in entities_map:
                entities_map[key] = (entity, [])
            if getattr(entity, "score", None) is not None:
                entities_map[key][1].append(entity.score)

        return [
            entity.model_copy(update={"score": _average_scores(scores)}) for entity, scores in entities_map.values()
        ]


class SingleEntityConsolidation(ConsolidationStrategy):
    """Consolidation strategy for a single entity."""

    def __init__(self, extractor: Callable[[Any], pydantic.BaseModel | None]):
        """Initialize SingleEntityConsolidation.

        :param extractor: Callable to extract a single entity from a raw chunk result.
        """
        self._extractor = extractor

    def consolidate(
        self,
        results: Sequence[Any],
        docs_offsets: list[tuple[int, int]],
    ) -> Sequence[pydantic.BaseModel | None]:
        """Consolidate single entities from chunks via majority vote.

        :param results: Raw chunk results.
        :param docs_offsets: Chunk offsets per document.
        :return: Winner entity per document.
        """
        consolidated_results: list[pydantic.BaseModel | None] = []
        for start, end in docs_offsets:
            chunk_results = [
                (self._extractor(res) if res is not None else None, i) for i, res in enumerate(results[start:end])
            ]
            consolidated_results.append(self._consolidate_single(chunk_results))
        return consolidated_results

    @staticmethod
    def _consolidate_single(entities_with_indices: list[tuple[_EntityType | None, int]]) -> _EntityType | None:
        """Majority vote for single entity with tie-breaking based on first occurrence."""
        if not entities_with_indices:
            return None

        counts: Counter[str] = Counter()
        key_to_entity: dict[str, _EntityType | None] = {}
        first_seen: dict[str, int] = {}
        scores_map: dict[str, list[float]] = defaultdict(list)

        for entity, i in entities_with_indices:
            key = _get_entity_key(entity) if entity is not None else "null"
            counts[key] += 1
            if key not in key_to_entity:
                key_to_entity[key] = entity
            if key not in first_seen:
                first_seen[key] = i
            if entity is not None and getattr(entity, "score", None) is not None:
                scores_map[key].append(entity.score)

        max_count = max(counts.values())
        winners = [k for k, count in counts.items() if count == max_count]
        winner_key = min(winners, key=lambda k: first_seen[k])

        winner_entity = key_to_entity[winner_key]
        if winner_entity is None:
            return None

        return winner_entity.model_copy(update={"score": _average_scores(scores_map[winner_key])})


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
        self,
        results: Sequence[Any],
        docs_offsets: list[tuple[int, int]],
    ) -> Sequence[list[tuple[str, float]]]:
        """Consolidate label scores from chunks by averaging them.

        :param results: Raw chunk results.
        :param docs_offsets: Chunk offsets per document.
        :return: Sorted list of (label, score) pairs per document.
        """
        consolidated_results: list[list[tuple[str, float]]] = []
        for start, end in docs_offsets:
            chunk_results = [res for res in results[start:end] if res is not None]
            num_chunks = end - start
            label_scores = {label: 0.0 for label in self.labels}

            for res in chunk_results:
                scores = self.extractor(res)
                for label, score in scores.items():
                    if label in label_scores:
                        label_scores[label] += max(0.0, min(float(score), 1.0))

            avg_scores = sorted(
                ((label, score / num_chunks) for label, score in label_scores.items()),
                key=lambda x: x[1],
                reverse=True,
            )
            consolidated_results.append(avg_scores)

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
        self,
        results: Sequence[Any],
        docs_offsets: list[tuple[int, int]],
    ) -> Sequence[tuple[str, float | None]]:
        """Consolidate text chunks by joining them and averaging scores.

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

            consolidated_results.append((self.joiner.join(texts).strip(), _average_scores(scores)))

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
        self,
        results: Sequence[Any],
        docs_offsets: list[tuple[int, int]],
    ) -> Sequence[list[tuple[str, str, float | None]]]:
        """Consolidate QA pairs using sequential matching.

        Assumes that the i-th QA pair returned by the model corresponds to the i-th question in self.questions.

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

                for i, (_, answer, score) in enumerate(self.extractor(res)):
                    if i < len(self.questions):
                        question = self.questions[i]
                        qa_map[question][0].append(answer)
                        if score is not None:
                            qa_map[question][1].append(score)

            consolidated_qa = [
                (q, " ".join(qa_map[q][0]).strip(), _average_scores(qa_map[q][1])) for q in self.questions
            ]
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
        self,
        results: Sequence[Any],
        docs_offsets: list[tuple[int, int]],
    ) -> Sequence[tuple[dict[str, float], float | None]]:
        """Consolidate map-based scores by averaging them.

        :param results: Raw chunk results.
        :param docs_offsets: Chunk offsets per document.
        :return: Consolidated (avg_map_scores, avg_overall_score) per document.
        """
        consolidated_results: list[tuple[dict[str, float], float | None]] = []
        for start, end in docs_offsets:
            num_chunks = end - start
            key_scores = {k: 0.0 for k in self.keys}
            overall_scores: list[float] = []

            for res in results[start:end]:
                if res is None:
                    continue
                m_scores, o_score = self.extractor(res)
                for k, s in m_scores.items():
                    if k in key_scores:
                        key_scores[k] += max(0.0, min(float(s), 1.0))
                if o_score is not None:
                    overall_scores.append(o_score)

            avg_map = {k: s / num_chunks for k, s in key_scores.items()}
            consolidated_results.append((avg_map, _average_scores(overall_scores)))

        return consolidated_results
