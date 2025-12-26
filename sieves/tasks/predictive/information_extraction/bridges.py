"""Bridges for information extraction task."""

import abc
from collections.abc import Callable, Iterable, Sequence
from functools import cached_property
from typing import Any, Literal, TypeVar, override

import dspy
import jinja2
import pydantic

from sieves.data import Doc
from sieves.model_wrappers import ModelWrapperInferenceMode, dspy_, langchain_, outlines_
from sieves.model_wrappers.types import ModelSettings
from sieves.tasks.predictive.bridges import Bridge
from sieves.tasks.predictive.consolidation import (
    MultiEntityConsolidation,
    SingleEntityConsolidation,
)
from sieves.tasks.predictive.schemas.information_extraction import ResultMulti, ResultSingle

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class InformationExtractionBridge(
    Bridge[_BridgePromptSignature, _BridgeResult, ModelWrapperInferenceMode],
    abc.ABC,
):
    """Abstract base class for information extraction bridges."""

    def __init__(
        self,
        task_id: str,
        prompt_instructions: str | None,
        entity_type: type[pydantic.BaseModel],
        model_settings: ModelSettings,
        mode: Literal["multi", "single"] = "multi",
    ):
        """Initialize InformationExtractionBridge.

        :param task_id: Task ID.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param entity_type: Type to extract.
        :param model_settings: Model settings including inference_mode.
        :param mode: Extraction mode.
        """
        super().__init__(
            task_id=task_id,
            prompt_instructions=prompt_instructions,
            overwrite=False,
            model_settings=model_settings,
        )
        self._entity_type = entity_type
        self._mode = mode
        if self._mode == "multi":
            self._consolidation_strategy = MultiEntityConsolidation(extractor=self._get_multi_extractor())
        else:
            self._consolidation_strategy = SingleEntityConsolidation(extractor=self._get_single_extractor())

    @abc.abstractmethod
    def _get_multi_extractor(self) -> Callable[[Any], Iterable[pydantic.BaseModel]]:
        """Return a callable that extracts a list of entities from a raw chunk result.

        :return: Multi-extractor callable.
        """

    @abc.abstractmethod
    def _get_single_extractor(self) -> Callable[[Any], pydantic.BaseModel | None]:
        """Return a callable that extracts a single entity from a raw chunk result.

        :return: Single-extractor callable.
        """


class DSPyInformationExtraction(InformationExtractionBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    """DSPy bridge for information extraction."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        if self._mode == "multi":
            return (
                "Find all occurences of this kind of entitity within the text. For each entity found, also provide "
                "a confidence score between 0.0 and 1.0 in the 'score' field. "
                "The `score` field must be provided regardless of whether or not potentially given fewshot examples "
                "have it."
            )
        return (
            "Find the single most relevant entitity within the text. If no such entitity exists, return null. Return "
            "exactly one entity with all its fields, NOT just a string. Also provide a confidence score between 0.0 "
            "and 1.0 in the 'score' field."
            "The `score` field must be provided regardless of whether or not potentially given fewshot examples "
            "have it."
        )

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

        if self._mode == "multi":

            class InformationExtractionMulti(dspy.Signature):  # type: ignore[misc]
                text: str = dspy.InputField(description="Text to extract entities from.")
                entities: list[extraction_type] = dspy.OutputField(description="Entities to extract from text.")  # type: ignore[valid-type]

            cls = InformationExtractionMulti
        else:

            class InformationExtractionSingle(dspy.Signature):  # type: ignore[misc]
                text: str = dspy.InputField(description="Text to extract entity from.")
                entity: extraction_type | None = dspy.OutputField(  # type: ignore[valid-type]
                    description="The single entity to extract from text. Should NOT be a list."
                )

            cls = InformationExtractionSingle

        cls.__doc__ = jinja2.Template(self._prompt_instructions).render()

        return cls

    @override
    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return self._model_settings.inference_mode or dspy_.InferenceMode.predict

    @override
    def _get_multi_extractor(self) -> Callable[[Any], Iterable[pydantic.BaseModel]]:
        return lambda res: res.entities

    @override
    def _get_single_extractor(self) -> Callable[[Any], pydantic.BaseModel | None]:
        return lambda res: res.entity

    @override
    def integrate(self, results: Sequence[dspy_.Result], docs: list[Doc]) -> list[Doc]:
        for doc, result in zip(docs, results):
            if self._mode == "multi":
                assert len(result.completions.entities) == 1
                doc.results[self._task_id] = ResultMulti(
                    entities=result.completions.entities[0],
                )
            else:
                assert len(result.completions.entity) == 1
                doc.results[self._task_id] = ResultSingle(
                    entity=result.completions.entity[0],
                )
        return docs

    @override
    def consolidate(
        self, results: Sequence[dspy_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[dspy_.Result]:
        consolidated_results_clean = self._consolidation_strategy.consolidate(results, docs_offsets)

        # Wrap back into dspy.Prediction.
        consolidated_results: list[dspy_.Result] = []
        for res_clean in consolidated_results_clean:
            if self._mode == "multi":
                consolidated_results.append(
                    dspy.Prediction.from_completions(
                        {"entities": [res_clean]},
                        signature=self.prompt_signature,
                    )
                )
            else:
                consolidated_results.append(
                    dspy.Prediction.from_completions(
                        {"entity": [res_clean]},
                        signature=self.prompt_signature,
                    )
                )
        return consolidated_results


class PydanticBasedInformationExtraction(
    InformationExtractionBridge[pydantic.BaseModel, pydantic.BaseModel, ModelWrapperInferenceMode],
    abc.ABC,
):
    """Base class for Pydantic-based information extraction bridges."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        if self._mode == "multi":
            return """
            Find all occurences of this kind of entitity within the text. For each entity found, also provide
            a confidence score between 0.0 and 1.0 in the 'score' field.
            """
        return """
        Find the single most relevant entitity within the text. If no such entitity exists, return null. Return exactly
        one entity with all its fields, NOT just a string. Also provide a confidence score between 0.0 and 1.0 in the
        'score' field.
        """

    @override
    @property
    def _prompt_example_template(self) -> str | None:
        if self._mode == "multi":
            return """
            {% if examples|length > 0 -%}
                <examples>
                {%- for example in examples %}
                    <example>
                        <text>{{ example.text }}</text>
                        <output>
                            <entities>{{ example.entities }}</entities>
                        </output>
                    </example>
                {% endfor -%}
                </examples>
            {% endif -%}
            """
        return """
        {% if examples|length > 0 -%}
            <examples>
            {%- for example in examples %}
                <example>
                    <text>{{ example.text }}</text>
                    <output>
                        <entity>{{ example.entity }}</entity>
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

        if self._mode == "multi":

            class InformationExtractionMulti(pydantic.BaseModel, frozen=True):
                """Entities to extract from text."""

                entities: list[entity_type]  # type: ignore[valid-type]

            return InformationExtractionMulti
        else:

            class InformationExtractionSingle(pydantic.BaseModel, frozen=True):
                """Entity to extract from text."""

                entity: entity_type | None = pydantic.Field(  # type: ignore[valid-type]
                    description="The single entity to extract from text. Should NOT be a list."
                )

            return InformationExtractionSingle

    @override
    def _get_multi_extractor(self) -> Callable[[Any], Iterable[pydantic.BaseModel]]:
        return lambda res: res.entities

    @override
    def _get_single_extractor(self) -> Callable[[Any], pydantic.BaseModel | None]:
        return lambda res: res.entity

    @override
    def integrate(self, results: Sequence[pydantic.BaseModel], docs: list[Doc]) -> list[Doc]:
        for doc, result in zip(docs, results):
            if self._mode == "multi":
                assert hasattr(result, "entities")
                doc.results[self._task_id] = ResultMulti(entities=result.entities)
            else:
                assert hasattr(result, "entity")
                doc.results[self._task_id] = ResultSingle(entity=result.entity)
        return docs

    @override
    def consolidate(
        self,
        results: Sequence[pydantic.BaseModel],
        docs_offsets: list[tuple[int, int]],  # type: ignore[arg-type]
    ) -> Sequence[pydantic.BaseModel]:
        consolidated_results_clean = self._consolidation_strategy.consolidate(results, docs_offsets)

        consolidated_results: list[pydantic.BaseModel] = []
        for res_clean in consolidated_results_clean:
            if self._mode == "multi":
                consolidated_results.append(self.prompt_signature(entities=res_clean))
            else:
                consolidated_results.append(self.prompt_signature(entity=res_clean))
        return consolidated_results


class OutlinesInformationExtraction(PydanticBasedInformationExtraction[outlines_.InferenceMode]):
    """Outlines bridge for information extraction."""

    @override
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return self._model_settings.inference_mode or outlines_.InferenceMode.json


class LangChainInformationExtraction(PydanticBasedInformationExtraction[langchain_.InferenceMode]):
    """LangChain bridge for information extraction."""

    @override
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return self._model_settings.inference_mode or langchain_.InferenceMode.structured
