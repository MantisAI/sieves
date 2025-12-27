"""Bridges for summarization task."""

import abc
from collections.abc import Callable, Sequence
from typing import Any, TypeVar, override

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
from sieves.tasks.predictive.consolidation import TextConsolidation
from sieves.tasks.predictive.schemas.summarization import Result

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class SummarizationBridge(Bridge[_BridgePromptSignature, _BridgeResult, ModelWrapperInferenceMode], abc.ABC):
    """Abstract base class for summarization bridges."""

    def __init__(
        self,
        task_id: str,
        prompt_instructions: str | None,
        n_words: int,
        overwrite: bool,
        model_settings: ModelSettings,
        prompt_signature: type[pydantic.BaseModel],
    ):
        """Initialize summarization bridge.

        :param task_id: Task ID.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param n_words: Maximum number of words for summary.
        :param overwrite: Whether to overwrite original text with summary.
        :param model_settings: Settings for structured generation.
        :param prompt_signature: Unified Pydantic prompt signature.
        """
        super().__init__(
            task_id=task_id,
            prompt_instructions=prompt_instructions,
            overwrite=overwrite,
            model_settings=model_settings,
            prompt_signature=prompt_signature,
        )
        self._n_words = n_words
        self._consolidation_strategy = TextConsolidation(extractor=self._chunk_extractor)

    @property
    @abc.abstractmethod
    def _chunk_extractor(self) -> Callable[[Any], tuple[str, float | None]]:
        """Return a callable that extracts (text, score) from a raw chunk result.

        :return: Extractor callable.
        """


class DSPySummarization(SummarizationBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    """DSPy bridge for summarization."""

    @override
    @property
    def model_type(self) -> ModelType:
        return ModelType.dspy

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return ""

    @override
    @property
    def _prompt_example_template(self) -> str | None:
        return None

    @override
    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return self._model_settings.inference_mode or dspy_.InferenceMode.predict

    @property
    @override
    def _chunk_extractor(self) -> Callable[[Any], tuple[str, float | None]]:
        return lambda res: (res.summary, getattr(res, "score", None))

    @override
    def integrate(self, results: Sequence[dspy_.Result], docs: list[Doc]) -> list[Doc]:
        for doc, result in zip(docs, results):
            assert len(result.completions.summary) == 1
            res = Result(summary=result.summary, score=getattr(result, "score", None))
            doc.results[self._task_id] = res

            if self._overwrite:
                doc.text = res.summary

        return docs

    @override
    def consolidate(
        self, results: Sequence[dspy_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[dspy_.Result]:
        consolidated_results_clean = self._consolidation_strategy.consolidate(results, docs_offsets)

        # Wrap back into dspy.Prediction.
        consolidated_results: list[dspy_.Result] = []
        for summary, score in consolidated_results_clean:
            consolidated_results.append(
                dspy.Prediction.from_completions(
                    {
                        "summary": [summary],
                        "score": [score],
                    },
                    signature=self.prompt_signature,
                )
            )

        return consolidated_results


class PydanticBasedSummarization(
    SummarizationBridge[pydantic.BaseModel, pydantic.BaseModel, ModelWrapperInferenceMode],
    abc.ABC,
):
    """Base class for Pydantic-based summarization bridges."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return "Your goal is to summarize a text. Provide a confidence score between 0.0 and 1.0 for the summary."

    @override
    @property
    def _prompt_example_template(self) -> str:
        return """
        {% if examples|length > 0 -%}
            <examples>
            {%- for example in examples %}
                <text>{{ example.text }}</text>
                <approximate_number_of_words_in_summary>{{ example.n_words }}</approximate_number_of_words_in_summary>
                <summary>
                {{ example.summary }}
                </summary>
                <score>{{ example.score }}</score>
            {% endfor -%}
            </examples>
        {% endif -%}
        """

    @override
    @property
    def _prompt_conclusion(self) -> str:
        return """
        ========
        <text>{{ text }}</text>
        <summary>
        """

    @property
    @override
    def _chunk_extractor(self) -> Callable[[Any], tuple[str, float | None]]:
        return lambda res: (res.summary, getattr(res, "score", None))

    @override
    def integrate(self, results: Sequence[pydantic.BaseModel], docs: list[Doc]) -> list[Doc]:
        for doc, result in zip(docs, results):
            assert hasattr(result, "summary")
            res = Result(summary=result.summary, score=getattr(result, "score", None))
            doc.results[self._task_id] = res

            if self._overwrite:
                doc.text = res.summary
        return docs

    @override
    def consolidate(
        self, results: Sequence[pydantic.BaseModel], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[pydantic.BaseModel]:
        assert issubclass(self.prompt_signature, pydantic.BaseModel)

        consolidated_results_clean = self._consolidation_strategy.consolidate(results, docs_offsets)
        consolidated_results: list[pydantic.BaseModel] = []

        for summary, score in consolidated_results_clean:
            consolidated_results.append(
                self.prompt_signature(
                    summary=summary,
                    score=score,
                )
            )

        return consolidated_results


class OutlinesSummarization(PydanticBasedSummarization[outlines_.InferenceMode]):
    """Outlines bridge for summarization."""

    @override
    @property
    def model_type(self) -> ModelType:
        return ModelType.outlines

    @override
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return self._model_settings.inference_mode or outlines_.InferenceMode.json


class LangChainSummarization(PydanticBasedSummarization[langchain_.InferenceMode]):
    """LangChain bridge for summarization."""

    @override
    @property
    def model_type(self) -> ModelType:
        return ModelType.langchain

    @override
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return self._model_settings.inference_mode or langchain_.InferenceMode.structured
