"""Bridges for information extraction task."""

import abc
from collections.abc import Iterable
from functools import cached_property
from typing import TypeVar, override

import dspy
import jinja2
import pydantic

from sieves.data import Doc
from sieves.engines import EngineInferenceMode, dspy_, langchain_, outlines_
from sieves.engines.types import GenerationSettings
from sieves.tasks.predictive.bridges import Bridge

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class InformationExtractionBridge(
    Bridge[_BridgePromptSignature, _BridgeResult, EngineInferenceMode],
    abc.ABC,
):
    """Abstract base class for information extraction bridges."""

    def __init__(
        self,
        task_id: str,
        prompt_instructions: str | None,
        entity_type: type[pydantic.BaseModel],
        generation_settings: GenerationSettings,
    ):
        """Initialize InformationExtractionBridge.

        :param task_id: Task ID.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param entity_type: Type to extract.
        :param generation_settings: Generation settings including inference_mode.
        """
        super().__init__(
            task_id=task_id,
            prompt_instructions=prompt_instructions,
            overwrite=False,
            generation_settings=generation_settings,
        )
        self._entity_type = entity_type


class DSPyInformationExtraction(InformationExtractionBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    """DSPy bridge for information extraction."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return "Find all occurences of this kind of entitity within the text."

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
        extraction_type = self._entity_type

        class Entities(dspy.Signature):  # type: ignore[misc]
            text: str = dspy.InputField(description="Text to extract entities from.")
            reasoning: str | None = dspy.OutputField(
                default=None, description="Provide reasoning for complex extraction cases."
            )
            entities: list[extraction_type] = dspy.OutputField(description="Entities to extract from text.")  # type: ignore[valid-type]

        Entities.__doc__ = jinja2.Template(self._prompt_instructions).render()

        return Entities

    @override
    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return self._generation_settings.inference_mode or dspy_.InferenceMode.predict

    @override
    def integrate(self, results: Iterable[dspy_.Result], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            assert len(result.completions.entities) == 1
            doc.results[self._task_id] = result.completions.entities[0]
        return docs

    @override
    def consolidate(
        self, results: Iterable[dspy_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[dspy_.Result]:
        results = list(results)
        entity_type = self._entity_type
        entity_type_is_frozen = entity_type.model_config.get("frozen", False)

        # Merge all found entities.
        for doc_offset in docs_offsets:
            reasonings: list[str] = []
            entities: list[entity_type] = []  # type: ignore[valid-type]
            seen_entities: set[entity_type] = set()  # type: ignore[valid-type]

            for res in results[doc_offset[0] : doc_offset[1]]:
                if res is None:
                    continue
                reasonings.append(res.reasoning or "")
                assert len(res.completions.entities) == 1
                if entity_type_is_frozen:
                    # Ensure not to add duplicate entities.
                    for entity in res.completions.entities[0]:
                        if entity not in seen_entities:
                            entities.append(entity)
                            seen_entities.add(entity)
                else:
                    entities.extend(res.completions.entities[0])

            yield dspy.Prediction.from_completions(
                {"entities": [entities], "reasoning": [str(reasonings)]},
                signature=self.prompt_signature,
            )


class PydanticBasedInformationExtraction(
    InformationExtractionBridge[pydantic.BaseModel, pydantic.BaseModel, EngineInferenceMode],
    abc.ABC,
):
    """Base class for Pydantic-based information extraction bridges."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return """
        Find all occurences of this kind of entitity within the text. Keep your reasoning concise - don't
        exhaustively list all identified entities in your reasoning.
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
                    <output>
                        <reasoning>{{ example.reasoning }}</reasoning>
                        <entities>{{ example.entities }}</entities>
                    </output>
                </example>
            {% endfor -%}
            </examples>
        {% endif -%}
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
        entity_type = self._entity_type

        class Entity(pydantic.BaseModel, frozen=True):
            """Entity to extract from text."""

            reasoning: str | None = pydantic.Field(
                default=None, description="Provide reasoning for complex extraction cases."
            )
            entities: list[entity_type]  # type: ignore[valid-type]

        return Entity

    @override
    def integrate(self, results: Iterable[pydantic.BaseModel], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            assert hasattr(result, "entities")
            doc.results[self._task_id] = result.entities
        return docs

    @override
    def consolidate(
        self, results: Iterable[pydantic.BaseModel], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[pydantic.BaseModel]:
        results = list(results)
        entity_type = self._entity_type
        entity_type_is_frozen = entity_type.model_config.get("frozen", False)

        # Determine label scores for chunks per document.
        for doc_offset in docs_offsets:
            reasonings: list[str] = []
            entities: list[entity_type] = []  # type: ignore[valid-type]
            seen_entities: set[entity_type] = set()  # type: ignore[valid-type]

            for res in results[doc_offset[0] : doc_offset[1]]:
                if res is None:
                    continue  # type: ignore[unreachable]

                assert hasattr(res, "reasoning")
                reasonings.append(res.reasoning or "")

                assert hasattr(res, "entities")
                if entity_type_is_frozen:
                    # Ensure not to add duplicate entities.
                    for entity in res.entities:
                        if entity not in seen_entities:
                            entities.append(entity)
                            seen_entities.add(entity)
                else:
                    entities.extend(res.entities)

            yield self.prompt_signature(entities=entities, reasoning=str(reasonings))


class OutlinesInformationExtraction(PydanticBasedInformationExtraction[outlines_.InferenceMode]):
    """Outlines bridge for information extraction."""

    @override
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return self._generation_settings.inference_mode or outlines_.InferenceMode.json


class LangChainInformationExtraction(PydanticBasedInformationExtraction[langchain_.InferenceMode]):
    """LangChain bridge for information extraction."""

    @override
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return self._generation_settings.inference_mode or langchain_.InferenceMode.structured
