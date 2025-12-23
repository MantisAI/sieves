"""Bridges for relation extraction task."""

from __future__ import annotations

import abc
from collections.abc import Sequence
from functools import cached_property
from typing import Any, Literal, TypeVar, override

import dspy
import jinja2
import pydantic

from sieves.data import Doc
from sieves.model_wrappers import (
    ModelWrapperInferenceMode,
    dspy_,
    langchain_,
    outlines_,
)
from sieves.model_wrappers.types import ModelSettings
from sieves.tasks.predictive.bridges import Bridge
from sieves.tasks.predictive.schemas.relation_extraction import (
    RelationEntity,
    RelationEntityWithContext,
    RelationTriplet,
    RelationTripletWithContext,
    Result,
)

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class RelationExtractionBridge(Bridge[_BridgePromptSignature, _BridgeResult, ModelWrapperInferenceMode], abc.ABC):
    """Abstract base class for relation extraction bridges."""

    def __init__(
        self,
        task_id: str,
        relations: list[str] | dict[str, str],
        entity_types: list[str] | dict[str, str] | None,
        prompt_instructions: str | None,
        model_settings: ModelSettings,
    ):
        """Initialize RelationExtractionBridge.

        :param task_id: Task ID.
        :param relations: Relation types to extract.
        :param entity_types: Entity types to consider.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param model_settings: Model settings.
        """
        super().__init__(
            task_id=task_id,
            prompt_instructions=prompt_instructions,
            overwrite=False,
            model_settings=model_settings,
        )

        if isinstance(relations, dict):
            self._relations: list[str] = list(relations.keys())
            self._relation_descriptions: dict[str, str] = relations
        else:
            self._relations: list[str] = relations
            self._relation_descriptions: dict[str, str] = {}

        self._entity_types: list[str] | None = None
        self._entity_type_descriptions: dict[str, str] = {}

        if isinstance(entity_types, dict):
            self._entity_types = list(entity_types.keys())
            self._entity_type_descriptions = entity_types
        elif entity_types is not None:
            self._entity_types = entity_types
            self._entity_type_descriptions: dict[str, str] = {}

    def _get_relation_descriptions(self) -> str:
        """Return relation descriptions as a string.

        :return: Relation descriptions.
        """
        descs: list[str] = []
        for rel in self._relations:
            if rel in self._relation_descriptions:
                descs.append(
                    f"<relation_description><relation>{rel}</relation><description>"
                    f"{self._relation_descriptions[rel]}</description></relation_description>"
                )
            else:
                descs.append(rel)
        return "\n\t\t\t".join(descs)

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
                    f"<entity_type_description><type>{et}</type><description>"
                    f"{self._entity_type_descriptions[et]}</description></entity_type_description>"
                )
            else:
                descs.append(et)

        return "\n\t\t\t".join(descs)

    def _get_dynamic_relation_triple_model(self) -> type[pydantic.BaseModel]:
        """Create dynamic model for triplets with strict type constraints.

        :return: Triplet model.
        """
        AllowedEntityType = Literal[*self._entity_types] if self._entity_types else str  # type: ignore[valid-type]
        AllowedRelationType = Literal[*self._relations] if self._relations else str  # type: ignore[valid-type]

        class _RelationEntityWithContext(pydantic.BaseModel):
            text: str
            context: str
            entity_type: AllowedEntityType

        class _RelationTripletWithContext(pydantic.BaseModel):
            head: _RelationEntityWithContext
            relation: AllowedRelationType
            tail: _RelationEntityWithContext

        _RelationEntityWithContext.__doc__ = RelationEntityWithContext.__doc__
        _RelationTripletWithContext.__doc__ = RelationTripletWithContext.__doc__

        return _RelationTripletWithContext

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
    ) -> Sequence[list[Any]]:
        consolidated: list[list[Any]] = []

        for start, end in docs_offsets:
            doc_results = results[start:end]
            all_triplets: list[Any] = []
            seen: set[tuple[str, str, str]] = set()

            for res in doc_results:
                if res and hasattr(res, "triplets"):
                    for triplet in res.triplets:
                        # Use a simple key for deduplication within the bridge's internal format.
                        key = (getattr(triplet.head, "text", ""), triplet.relation, getattr(triplet.tail, "text", ""))
                        if key not in seen:
                            all_triplets.append(triplet)
                            seen.add(key)

            consolidated.append(all_triplets)

        return consolidated


class DSPyRelationExtraction(RelationExtractionBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    """DSPy bridge for relation extraction."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return f"""
        Extract relations between entities in the text.
        Relations to look for: {self._relations}
        {self._get_relation_descriptions()}
        Entity types to consider: {self._entity_types or "Unbounded"}
        {self._get_entity_type_descriptions()}

        For each triplet:
        - head: the subject entity (text, type)
        - relation: the type of relation
        - tail: the object entity (text, type)
        """

    @override
    @property
    def _prompt_example_template(self) -> str | None:
        return None

    @override
    @property
    def _prompt_conclusion(self) -> str | None:
        return None

    @override
    @cached_property
    def prompt_signature(self) -> type[dspy_.PromptSignature]:
        _RelationTripletWithContext = self._get_dynamic_relation_triple_model()

        class RelationExtraction(dspy.Signature):
            text: str = dspy.InputField()
            triplets: list[_RelationTripletWithContext] = dspy.OutputField()  # type: ignore[valid-type]

        RelationExtraction.__doc__ = jinja2.Template(self._prompt_instructions).render()
        RelationExtraction.model_rebuild()

        return RelationExtraction

    @override
    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return self._model_settings.inference_mode or dspy_.InferenceMode.predict


class PydanticBasedRelationExtraction(
    RelationExtractionBridge[pydantic.BaseModel, pydantic.BaseModel | list[Any], ModelWrapperInferenceMode], abc.ABC
):
    """Base class for Pydantic-based relation extraction bridges."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return f"""
        Extract relations between entities in the text.
        Relations: {self._relations}
        Entity Types: {self._entity_types or "Any"}
        Return a list of triplets with head, relation, and tail.
        """

    @override
    @property
    def _prompt_example_template(self) -> str | None:
        return """
        {% if examples|length > 0 -%}
        <examples>
        {%- for example in examples %}
            <example>
                <text>{{ example.text }}</text>
                <output>{{ example.triplets }}</output>
            </example>
        {% endfor -%}
        </examples>
        {% endif %}
        """

    @override
    @property
    def _prompt_conclusion(self) -> str | None:
        return """
        ========
        <text>{{ text }}</text>
        <output>
        """

    @override
    @cached_property
    def prompt_signature(self) -> type[pydantic.BaseModel]:
        _RelationTripletWithContext = self._get_dynamic_relation_triple_model()

        class RelationExtraction(pydantic.BaseModel):
            triplets: list[_RelationTripletWithContext]  # type: ignore[valid-type]

        return RelationExtraction


class OutlinesRelationExtraction(PydanticBasedRelationExtraction[outlines_.InferenceMode]):
    """Outlines bridge for relation extraction."""

    @override
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return self._model_settings.inference_mode or outlines_.InferenceMode.json


class LangChainRelationExtraction(PydanticBasedRelationExtraction[langchain_.InferenceMode]):
    """LangChain bridge for relation extraction."""

    @override
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return self._model_settings.inference_mode or langchain_.InferenceMode.structured
