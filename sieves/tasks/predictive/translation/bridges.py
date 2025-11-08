"""Bridges for translation task."""

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


class TranslationBridge(
    Bridge[_BridgePromptSignature, _BridgeResult, EngineInferenceMode],
    abc.ABC,
):
    """Abstract base class for translation bridges."""

    def __init__(
        self,
        task_id: str,
        prompt_instructions: str | None,
        overwrite: bool,
        language: str,
        generation_settings: GenerationSettings,
    ):
        """Initialize TranslationBridge.

        :param task_id: Task ID.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param overwrite: Whether to overwrite text with translation.
        :param language: Language to translate to.
        :param generation_settings: Generation settings including inference_mode.
        """
        super().__init__(
            task_id=task_id,
            prompt_instructions=prompt_instructions,
            overwrite=overwrite,
            generation_settings=generation_settings,
        )
        self._to = language

    @override
    def extract(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        return ({"text": doc.text if doc.text else None, "target_language": self._to} for doc in docs)


class DSPyTranslation(TranslationBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    """DSPy bridge for translation."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return "Translate this text into the target language."

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
        class Translation(dspy.Signature):  # type: ignore[misc]
            text: str = dspy.InputField()
            target_language: str = dspy.InputField()
            reasoning: str | None = dspy.OutputField(
                default=None, description="Provide reasoning for translation choices when relevant."
            )
            translation: str = dspy.OutputField()

        Translation.__doc__ = jinja2.Template(self._prompt_instructions).render()

        return Translation

    @override
    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return self._generation_settings.inference_mode or dspy_.InferenceMode.predict

    @override
    def integrate(self, results: Iterable[dspy_.Result], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            assert len(result.completions.translation) == 1
            doc.results[self._task_id] = result.translation

            if self._overwrite:
                doc.text = result.translation
        return docs

    @override
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
    """Base class for Pydantic-based translation bridges."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return """
        Translate into {{ target_language }}.
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
                    <target_language>{{ example.to }}</target_language>
                    <translation>
                    {{ example.translation }}
                    </translation>
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
        <target_language>{{ target_language }}</target_language>
        <translation>
        """

    @override
    @cached_property
    def prompt_signature(self) -> type[pydantic.BaseModel]:
        class Translation(pydantic.BaseModel, frozen=True):
            """Translation."""

            reasoning: str | None = pydantic.Field(
                default=None, description="Provide reasoning for translation choices when relevant."
            )
            translation: str

        return Translation

    @override
    def integrate(self, results: Iterable[pydantic.BaseModel], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            assert hasattr(result, "translation")
            doc.results[self._task_id] = result.translation

            if self._overwrite:
                doc.text = result.translation
        return docs

    @override
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
    """Outlines bridge for translation."""

    @override
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return self._generation_settings.inference_mode or outlines_.InferenceMode.json


class LangChainTranslation(PydanticBasedTranslation[langchain_.InferenceMode]):
    """LangChain bridge for translation."""

    @override
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return self._generation_settings.inference_mode or langchain_.InferenceMode.structured
