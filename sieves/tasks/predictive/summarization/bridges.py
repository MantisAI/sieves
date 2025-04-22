import abc
from collections.abc import Iterable
from functools import cached_property
from typing import Any, TypeVar

import dspy
import jinja2
import pydantic

from sieves.data import Doc
from sieves.engines import EngineInferenceMode, dspy_, instructor_, langchain_, ollama_, outlines_, vllm_
from sieves.tasks.predictive.bridges import Bridge

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class SummarizationBridge(
    Bridge[_BridgePromptSignature, _BridgeResult, EngineInferenceMode],
    abc.ABC,
):
    def __init__(
        self,
        task_id: str,
        prompt_template: str | None,
        prompt_signature_desc: str | None,
        overwrite: bool,
        n_words: int,
    ):
        """
        Initializes InformationExtractionBridge.
        :param task_id: Task ID.
        :param prompt_template: Custom prompt template.
        :param prompt_signature_desc: Custom prompt signature description.
        :param overwrite: Whether to overwrite text with summarization text.
        :param n_words: Approximate number of words in summary.
        """
        super().__init__(
            task_id=task_id,
            prompt_template=prompt_template,
            prompt_signature_desc=prompt_signature_desc,
            overwrite=overwrite,
        )
        self._n_words = n_words

    def extract(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        return ({"text": doc.text if doc.text else None, "n_words": self._n_words} for doc in docs)


class DSPySummarization(SummarizationBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    @property
    def _prompt_template(self) -> str | None:
        return None

    @property
    def _prompt_signature_description(self) -> str | None:
        return "Summary of a longer text."

    @cached_property
    def prompt_signature(self) -> type[dspy_.PromptSignature]:
        class Summary(dspy.Signature):  # type: ignore[misc]
            text: str = dspy.InputField(description="Text to summarize.")
            n_words: str = dspy.InputField(description="Number of words to approximately use for summary.")
            summary: str = dspy.OutputField(description="Summary of text.")

        Summary.__doc__ = jinja2.Template(self.prompt_signature_description).render()

        return Summary

    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return dspy_.InferenceMode.chain_of_thought

    def integrate(self, results: Iterable[dspy_.Result], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            assert len(result.completions.summary) == 1
            doc.results[self._task_id] = result.summary

            if self._overwrite:
                doc.text = result.summary

        return docs

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
    @property
    def _prompt_template(self) -> str | None:
        return """
        Your goal is to summarize a text. This summary should be around {{ max_n }} words.

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

        ========

        <text>{{ text }}</text>
        <approximate_number_of_words_in_summary>{{ n_words }}</approximate_number_of_words_in_summary>
        <summary> 
        """

    @property
    def _prompt_signature_description(self) -> str | None:
        return None

    @cached_property
    def prompt_signature(self) -> type[pydantic.BaseModel]:
        class Summary(pydantic.BaseModel, frozen=True):
            summary: str

        if self.prompt_signature_description:
            Summary.__doc__ = jinja2.Template(self.prompt_signature_description).render()

        return Summary

    def integrate(self, results: Iterable[pydantic.BaseModel], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            assert hasattr(result, "summary")
            doc.results[self._task_id] = result.summary

            if self._overwrite:
                doc.text = result.summary
        return docs

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
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return outlines_.InferenceMode.json


class OllamaSummarization(PydanticBasedSummarization[ollama_.InferenceMode]):
    @property
    def inference_mode(self) -> ollama_.InferenceMode:
        return ollama_.InferenceMode.structured


class LangChainSummarization(PydanticBasedSummarization[langchain_.InferenceMode]):
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return langchain_.InferenceMode.structured


class InstructorSummarization(PydanticBasedSummarization[instructor_.InferenceMode]):
    @property
    def inference_mode(self) -> instructor_.InferenceMode:
        return instructor_.InferenceMode.structured


class VLLMSummarization(PydanticBasedSummarization[vllm_.InferenceMode]):
    @property
    def inference_mode(self) -> vllm_.InferenceMode:
        return vllm_.InferenceMode.json
