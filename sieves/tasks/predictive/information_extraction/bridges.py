"""Bridges for information extraction task."""

import abc
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, TypeVar, override

import dspy
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
from sieves.tasks.predictive.consolidation import (
    MultiEntityConsolidation,
    SingleEntityConsolidation,
)
from sieves.tasks.predictive.schemas.information_extraction import ResultMulti, ResultSingle

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class InformationExtractionBridge(Bridge[_BridgePromptSignature, _BridgeResult, ModelWrapperInferenceMode], abc.ABC):
    """Abstract base class for information extraction bridges."""

    def __init__(
        self,
        task_id: str,
        prompt_instructions: str | None,
        entity_type: type[pydantic.BaseModel],
        model_settings: ModelSettings,
        mode: Literal["multi", "single"],
        prompt_signature: type[pydantic.BaseModel],
        model_type: ModelType,
        fewshot_examples: Sequence[pydantic.BaseModel] = (),
    ):
        """Initialize information extraction bridge.

        :param task_id: Task ID.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param entity_type: Object type to extract.
        :param model_settings: Settings for structured generation.
        :param mode: Extraction mode. If "multi", all occurrences of the entity are extracted. If "single", exactly one
            (or no) entity is extracted.
        :param prompt_signature: Unified Pydantic prompt signature.
        :param model_type: Model type.
        :param fewshot_examples: Few-shot examples.
        """
        assert model_type in {ModelType.dspy, ModelType.gliner, ModelType.langchain, ModelType.outlines}

        super().__init__(
            task_id=task_id,
            prompt_instructions=prompt_instructions,
            overwrite=False,
            model_settings=model_settings,
            prompt_signature=prompt_signature,
            model_type=model_type,
            fewshot_examples=fewshot_examples,
        )
        self._entity_type = entity_type
        self._mode = mode

        if self._mode == "multi":
            self._consolidation_strategy = MultiEntityConsolidation(extractor=self._get_multi_extractor())
        else:
            self._consolidation_strategy = SingleEntityConsolidation(extractor=self._get_single_extractor())

    @staticmethod
    def _get_multi_extractor() -> Callable[[Any], Iterable[pydantic.BaseModel]]:
        """Return a callable that extracts a list of entities from a raw chunk result.

        :return: Multi-extractor callable.
        """
        return lambda res: res.entities

    @staticmethod
    def _get_single_extractor() -> Callable[[Any], pydantic.BaseModel | None]:
        """Return a callable that extracts a single entity from a raw chunk result.

        :return: Single-extractor callable.
        """
        return lambda res: res.entity


class DSPyInformationExtraction(InformationExtractionBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    """DSPy bridge for information extraction."""

    @override
    def _validate(self) -> None:
        assert self._model_type == ModelType.dspy

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return ""

    @override
    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return self._model_settings.inference_mode or dspy_.InferenceMode.predict

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


class PydanticInformationExtraction(
    InformationExtractionBridge[pydantic.BaseModel, pydantic.BaseModel, ModelWrapperInferenceMode],
):
    """Base class for Pydantic-based information extraction bridges."""

    @override
    def _validate(self) -> None:
        assert self._model_type in {ModelType.langchain, ModelType.outlines}

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        if self._mode == "multi":
            return (
                "Find all occurences of this kind of entitity within the text. For each entity found, also provide "
                "a confidence score between 0.0 and 1.0 in the 'score' field."
            )

        return (
            "Find the single most relevant entitity within the text. If no such entitity exists, return null. "
            "Return exactly one entity with all its fields, NOT just a string. Also provide a confidence score "
            "between 0.0 and 1.0 in the 'score' field."
        )

    @override
    @property
    def _prompt_conclusion(self) -> str | None:
        return "========\n\n<text>{{ text }}</text>"

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
        docs_offsets: list[tuple[int, int]],
    ) -> Sequence[pydantic.BaseModel]:
        assert issubclass(self.prompt_signature, pydantic.BaseModel)

        consolidated_results_clean = self._consolidation_strategy.consolidate(results, docs_offsets)
        consolidated_results: list[pydantic.BaseModel] = []

        for res_clean in consolidated_results_clean:
            if self._mode == "multi":
                consolidated_results.append(self.prompt_signature(entities=res_clean))
            else:
                consolidated_results.append(self.prompt_signature(entity=res_clean))

        return consolidated_results

    @override
    @property
    def inference_mode(self) -> outlines_.InferenceMode | langchain_.InferenceMode:
        if self._model_type == ModelType.outlines:
            return self._model_settings.inference_mode or outlines_.InferenceMode.json

        elif self._model_type == ModelType.langchain:
            return self._model_settings.inference_mode or langchain_.InferenceMode.structured

        raise ValueError(f"Unsupported model type: {self._model_type}")
