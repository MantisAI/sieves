"""Bridges for summarization task."""

import abc
from collections.abc import Iterable
from functools import cached_property
from typing import Any, TypeVar, override

import dspy
import jinja2
import pydantic

from sieves.data import Doc
from sieves.engines import EngineInferenceMode, dspy_, langchain_, outlines_
from sieves.engines.types import GenerationSettings
from sieves.tasks.predictive.bridges import Bridge

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class SummarizationBridge(
    Bridge[_BridgePromptSignature, _BridgeResult, EngineInferenceMode],
    abc.ABC,
):
    """Abstract base class for summarization bridges."""

    def __init__(
        self,
        task_id: str,
        prompt_instructions: str | None,
        overwrite: bool,
        n_words: int,
        generation_settings: GenerationSettings,
    ):
        """Initialize SummarizationBridge.

        :param task_id: Task ID.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param overwrite: Whether to overwrite text with summarization text.
        :param n_words: Approximate number of words in summary.
        :param generation_settings: Generation settings including inference_mode.
        """
        super().__init__(
            task_id=task_id,
            prompt_instructions=prompt_instructions,
            overwrite=overwrite,
            generation_settings=generation_settings,
        )
        self._n_words = n_words

    @override
    def extract(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        return ({"text": doc.text if doc.text else None, "n_words": self._n_words} for doc in docs)


class DSPySummarization(SummarizationBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    """DSPy bridge for summarization."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return "Summary of a longer text."

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
        class Summary(dspy.Signature):  # type: ignore[misc]
            text: str = dspy.InputField(description="Text to summarize.")
            n_words: str = dspy.InputField(description="Number of words to approximately use for summary.")
            reasoning: str | None = dspy.OutputField(
                default=None, description="Provide reasoning for summarization choices when appropriate."
            )
            summary: str = dspy.OutputField(description="Summary of text.")

        Summary.__doc__ = jinja2.Template(self._prompt_instructions).render()

        return Summary

    @override
    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return self._generation_settings.inference_mode or dspy_.InferenceMode.predict

    @override
    def integrate(self, results: Iterable[dspy_.Result], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            assert len(result.completions.summary) == 1
            doc.results[self._task_id] = result.summary

            if self._overwrite:
                doc.text = result.summary

        return docs

    @override
    def consolidate(
        self, results: Iterable[dspy_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[dspy_.Result]:
        results = list(results)

        # Merge all chunk translations.
        for doc_offset in docs_offsets:
            summaries: list[str] = []

            for res in results[doc_offset[0] : doc_offset[1]]:
                if res is None:
                    continue
                summaries.append(res.summary)

            yield dspy.Prediction.from_completions(
                {"summary": ["\n".join(summaries)]},
                signature=self.prompt_signature,
            )


class PydanticBasedSummarization(
    SummarizationBridge[pydantic.BaseModel, pydantic.BaseModel, EngineInferenceMode],
    abc.ABC,
):
    """Base class for Pydantic-based summarization bridges."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return """
        Your goal is to summarize a text. This summary should be around {{ max_n }} words.
        """

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
        <approximate_number_of_words_in_summary>{{ n_words }}</approximate_number_of_words_in_summary>
        <summary>
        """

    @override
    @cached_property
    def prompt_signature(self) -> type[pydantic.BaseModel]:
        class Summary(pydantic.BaseModel, frozen=True):
            """Summary of the specified text."""

            reasoning: str | None = pydantic.Field(
                default=None, description="Provide reasoning for summarization choices when appropriate."
            )
            summary: str

        return Summary

    @override
    def integrate(self, results: Iterable[pydantic.BaseModel], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            assert hasattr(result, "summary")
            doc.results[self._task_id] = result.summary

            if self._overwrite:
                doc.text = result.summary
        return docs

    @override
    def consolidate(
        self, results: Iterable[pydantic.BaseModel], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[pydantic.BaseModel]:
        results = list(results)

        # Determine label scores for chunks per document.
        for doc_offset in docs_offsets:
            summaries: list[str] = []

            for res in results[doc_offset[0] : doc_offset[1]]:
                if res is None:
                    continue  # type: ignore[unreachable]

                assert hasattr(res, "summary")
                summaries.append(res.summary)

            yield self.prompt_signature(summary="\n".join(summaries).strip())


class OutlinesSummarization(PydanticBasedSummarization[outlines_.InferenceMode]):
    """Outlines bridge for summarization."""

    @override
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return self._generation_settings.inference_mode or outlines_.InferenceMode.json


class LangChainSummarization(PydanticBasedSummarization[langchain_.InferenceMode]):
    """LangChain bridge for summarization."""

    @override
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return self._generation_settings.inference_mode or langchain_.InferenceMode.structured
