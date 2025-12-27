"""Bridges for relation extraction task."""

from __future__ import annotations

import abc
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar, override

import pydantic

from sieves.data import Doc
from sieves.model_wrappers import (
    ModelType,
    ModelWrapperInferenceMode,
    dspy_,
    langchain_,
    outlines_,
)
from sieves.model_wrappers.types import ModelSettings
from sieves.tasks.predictive.bridges import Bridge
from sieves.tasks.predictive.consolidation import MultiEntityConsolidation
from sieves.tasks.predictive.schemas.relation_extraction import (
    RelationEntity,
    RelationTriplet,
    Result,
)
from sieves.tasks.predictive.utils import convert_to_signature

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class RelationExtractionBridge(Bridge[_BridgePromptSignature, _BridgeResult, ModelWrapperInferenceMode], abc.ABC):
    """Abstract base class for relation extraction bridges."""

    def __init__(
        self,
        task_id: str,
        relations: Sequence[str] | dict[str, str],
        entity_types: Sequence[str] | dict[str, str] | None,
        prompt_instructions: str | None,
        model_settings: ModelSettings,
        prompt_signature: type[pydantic.BaseModel],
        model_type: ModelType,
        fewshot_examples: Sequence[pydantic.BaseModel] = (),
    ):
        """Initialize relation extraction bridge.

        :param task_id: Task ID.
        :param relations: Relations to extract. Can be a list of relation types or a dict mapping types to descriptions.
        :param entity_types: Entity types constraints.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param model_settings: Settings for structured generation.
        :param prompt_signature: Unified Pydantic prompt signature.
        :param model_type: Model type.
        :param fewshot_examples: Few-shot examples.
        """
        super().__init__(
            task_id=task_id,
            prompt_instructions=prompt_instructions,
            overwrite=False,
            model_settings=model_settings,
            prompt_signature=prompt_signature,
            model_type=model_type,
            fewshot_examples=fewshot_examples,
        )
        if isinstance(relations, dict):
            self._relations = list(relations.keys())
            self._relation_descriptions = relations
        else:
            self._relations = list(relations)
            self._relation_descriptions = {}

        if isinstance(entity_types, dict):
            self._entity_types = list(entity_types.keys())
            self._entity_type_descriptions = entity_types
        elif entity_types is not None:
            self._entity_types = list(entity_types)
            self._entity_type_descriptions = {}
        else:
            self._entity_types = None
            self._entity_type_descriptions = {}

        self._consolidation_strategy = MultiEntityConsolidation(extractor=self._chunk_extractor)

    @override
    @property
    def prompt_signature(self) -> _BridgePromptSignature:
        return convert_to_signature(
            model_cls=self._pydantic_signature,
            model_type=self.model_type,
            mode="relations",
        )  # type: ignore[return-value]

    @property
    @abc.abstractmethod
    def _chunk_extractor(self) -> Callable[[Any], Iterable[pydantic.BaseModel]]:
        """Return a callable that extracts a list of entities from a raw chunk result.

        :return: Extractor callable.
        """

    def _get_relation_descriptions(self) -> str:
        """Return relation descriptions as a string.

        :return: Relation descriptions.
        """
        descs: list[str] = []
        for rel in self._relations:
            if rel in self._relation_descriptions:
                descs.append(
                    f"  <relation_description>\n    <relation>{rel}</relation>\n    <description>"
                    f"{self._relation_descriptions[rel]}</description>\n  </relation_description>"
                )
            else:
                descs.append(f"  <relation>{rel}</relation>")
        return "\n".join(descs)

    def _get_entity_type_descriptions(self) -> str:
        """Return entity type descriptions as a string.

        :return: Entity type descriptions.
        """
        if self._entity_types is None:
            return "Unbounded"

        descs: list[str] = []
        for et in self._entity_types:
            if et in self._entity_type_descriptions:
                descs.append(
                    f"  <entity_type_description>\n    <type>{et}</type>\n    <description>"
                    f"{self._entity_type_descriptions[et]}</description>\n  </entity_type_description>"
                )
            else:
                descs.append(f"  <type>{et}</type>")

        return "\n".join(descs)

    def _process_triplets(self, raw_triplets: list[Any]) -> list[RelationTriplet]:
        """Convert raw triplets from model to RelationTriplet objects.

        :param raw_triplets: Raw triplets from the model.
        :return: Processed RelationTriplet objects.
        """
        processed: list[RelationTriplet] = []
        for raw in raw_triplets:
            head_text = getattr(raw.head, "text", "")
            head_type = getattr(raw.head, "entity_type", "")

            tail_text = getattr(raw.tail, "text", "")
            tail_type = getattr(raw.tail, "entity_type", "")

            processed.append(
                RelationTriplet(
                    head=RelationEntity(text=head_text, entity_type=head_type),
                    relation=getattr(raw, "relation", ""),
                    tail=RelationEntity(text=tail_text, entity_type=tail_type),
                    score=getattr(raw, "score", None),
                )
            )

        return processed

    @override
    def integrate(self, results: Sequence[_BridgeResult | list[Any]], docs: list[Doc]) -> list[Doc]:
        for doc, result in zip(docs, results):
            # Handle both model result objects and raw lists from consolidation.
            raw_triplets = result if isinstance(result, list) else getattr(result, "triplets", [])
            doc.results[self._task_id] = Result(triplets=self._process_triplets(raw_triplets))

        return docs

    @override
    def consolidate(
        self,
        results: Sequence[_BridgeResult],
        docs_offsets: list[tuple[int, int]],
    ) -> Sequence[list[pydantic.BaseModel]]:
        return self._consolidation_strategy.consolidate(results, docs_offsets)


class DSPyRelationExtraction(RelationExtractionBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    """DSPy bridge for relation extraction."""

    @override
    def _validate(self) -> None:
        assert self._model_type == ModelType.dspy

    @property
    @override
    def _chunk_extractor(self) -> Callable[[Any], Iterable[pydantic.BaseModel]]:
        return lambda res: res.triplets

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return ""

    @override
    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return self._model_settings.inference_mode or dspy_.InferenceMode.predict


class PydanticRelationExtraction(
    RelationExtractionBridge[pydantic.BaseModel, pydantic.BaseModel | list[Any], ModelWrapperInferenceMode], abc.ABC
):
    """Base class for Pydantic-based relation extraction bridges."""

    @override
    def _validate(self) -> None:
        assert self._model_type in {ModelType.langchain, ModelType.outlines}

    @property
    @override
    def _chunk_extractor(self) -> Callable[[Any], Iterable[pydantic.BaseModel]]:
        return lambda res: res.triplets

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return (
            "Extract relations between entities in the text.\n"
            f"Relations: {self._relations}\n"
            f"Entity Types: {self._entity_types or 'Any'}\n"
            "Return a list of triplets with head, relation, tail, and a confidence score between 0.0 and 1.0."
        )

    @override
    @property
    def _prompt_conclusion(self) -> str | None:
        return "========\n<text>{{ text }}</text>"

    @override
    @property
    def model_type(self) -> ModelType:
        return self._model_type

    @override
    @property
    def inference_mode(self) -> outlines_.InferenceMode | langchain_.InferenceMode:
        if self._model_type == ModelType.outlines:
            return self._model_settings.inference_mode or outlines_.InferenceMode.json
        elif self._model_type == ModelType.langchain:
            return self._model_settings.inference_mode or langchain_.InferenceMode.structured

        raise ValueError(f"Unsupported model type: {self._model_type}")
