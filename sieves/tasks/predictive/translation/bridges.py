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


class TranslationBridge(
    Bridge[_BridgePromptSignature, _BridgeResult, EngineInferenceMode],
    abc.ABC,
):
    def __init__(
        self,
        task_id: str,
        prompt_template: str | None,
        prompt_signature_desc: str | None,
        overwrite: bool,
        language: str,
    ):
        """
        Initializes InformationExtractionBridge.
        :param task_id: Task ID.
        :param prompt_template: Custom prompt template.
        :param prompt_signature_desc: Custom prompt signature description.
        :param overwrite: Whether to overwrite text with translation.
        :param language: Language to translate to.
        """
        super().__init__(
            task_id=task_id,
            prompt_template=prompt_template,
            prompt_signature_desc=prompt_signature_desc,
            overwrite=overwrite,
        )
        self._to = language

    def extract(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        return ({"text": doc.text if doc.text else None, "target_language": self._to} for doc in docs)


class DSPyTranslation(TranslationBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    @property
    def _prompt_template(self) -> str | None:
        return None

    @property
    def _prompt_signature_description(self) -> str | None:
        return "Translate this text into the target language."

    @cached_property
    def prompt_signature(self) -> type[dspy_.PromptSignature]:
        class Translation(dspy.Signature):  # type: ignore[misc]
            text: str = dspy.InputField()
            target_language: str = dspy.InputField()
            translation: str = dspy.OutputField()

        Translation.__doc__ = jinja2.Template(self.prompt_signature_description).render()

        return Translation

    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return dspy_.InferenceMode.predict

    def integrate(self, results: Iterable[dspy_.Result], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            assert len(result.completions.translation) == 1
            doc.results[self._task_id] = result.translation

            if self._overwrite:
                doc.text = result.translation
        return docs

    def consolidate(
        self, results: Iterable[dspy_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[dspy_.Result]:
        results = list(results)

        # Merge all chunk translations.
        for doc_offset in docs_offsets:
            translations: list[str] = []

            for res in results[doc_offset[0] : doc_offset[1]]:
                if res is None:
                    continue
                translations.append(res.translation)

            yield dspy.Prediction.from_completions(
                {"translation": ["\n".join(translations)]},
                signature=self.prompt_signature,
            )


class PydanticBasedTranslation(
    TranslationBridge[pydantic.BaseModel, pydantic.BaseModel, EngineInferenceMode],
    abc.ABC,
):
    @property
    def _prompt_template(self) -> str | None:
        return """
        Translate into {{ target_language }}.

        {% if examples|length > 0 -%}
            <examples>
            {%- for example in examples %}
                <example>
                    <text>{{ example.text }}</text>
                    <target_language>{{ example.to }}</target_language>
                    <translation>
                    {{ example.translation }}
                    </translation>
                </example>
            {% endfor -%}
            </examples>
        {% endif -%}

        ========
        
        <text>{{ text }}</text>
        <target_language>{{ target_language }}</target_language>
        <translation> 
        """

    @property
    def _prompt_signature_description(self) -> str | None:
        return None

    @cached_property
    def prompt_signature(self) -> type[pydantic.BaseModel]:
        class Translation(pydantic.BaseModel, frozen=True):
            translation: str

        if self.prompt_signature_description:
            Translation.__doc__ = jinja2.Template(self.prompt_signature_description).render()

        return Translation

    def integrate(self, results: Iterable[pydantic.BaseModel], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            assert hasattr(result, "translation")
            doc.results[self._task_id] = result.translation

            if self._overwrite:
                doc.text = result.translation
        return docs

    def consolidate(
        self, results: Iterable[pydantic.BaseModel], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[pydantic.BaseModel]:
        results = list(results)

        # Determine label scores for chunks per document.
        for doc_offset in docs_offsets:
            translations: list[str] = []

            for res in results[doc_offset[0] : doc_offset[1]]:
                if res is None:
                    continue  # type: ignore[unreachable]

                assert hasattr(res, "translation")
                translations.append(res.translation)

            yield self.prompt_signature(translation="\n".join(translations))


class OutlinesTranslation(PydanticBasedTranslation[outlines_.InferenceMode]):
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return outlines_.InferenceMode.json


class OllamaTranslation(PydanticBasedTranslation[ollama_.InferenceMode]):
    @property
    def inference_mode(self) -> ollama_.InferenceMode:
        return ollama_.InferenceMode.structured


class LangChainTranslation(PydanticBasedTranslation[langchain_.InferenceMode]):
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return langchain_.InferenceMode.structured


class InstructorTranslation(PydanticBasedTranslation[instructor_.InferenceMode]):
    @property
    def inference_mode(self) -> instructor_.InferenceMode:
        return instructor_.InferenceMode.structured


class VLLMTranslation(PydanticBasedTranslation[vllm_.InferenceMode]):
    @property
    def inference_mode(self) -> vllm_.InferenceMode:
        return vllm_.InferenceMode.json
