"""Bridges for sentiment analysis task."""

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
from sieves.tasks.predictive.consolidation import MapScoreConsolidation
from sieves.tasks.predictive.schemas.sentiment_analysis import Result

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class SentimentAnalysisBridge(Bridge[_BridgePromptSignature, _BridgeResult, ModelWrapperInferenceMode], abc.ABC):
    """Abstract base class for sentiment analysis bridges."""

    def __init__(
        self,
        task_id: str,
        prompt_instructions: str | None,
        aspects: tuple[str, ...],
        model_settings: ModelSettings,
        prompt_signature: type[pydantic.BaseModel],
    ):
        """Initialize sentiment analysis bridge.

        :param task_id: Task ID.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param aspects: Aspects to analyze.
        :param model_settings: Settings for structured generation.
        :param prompt_signature: Unified Pydantic prompt signature.
        """
        super().__init__(
            task_id=task_id,
            prompt_instructions=prompt_instructions,
            overwrite=False,
            model_settings=model_settings,
            prompt_signature=prompt_signature,
        )
        self._aspects = aspects
        self._consolidation_strategy = MapScoreConsolidation(extractor=self._get_extractor(), keys=list(self._aspects))

    @abc.abstractmethod
    def _get_extractor(self) -> Callable[[Any], tuple[dict[str, float], float | None]]:
        """Return a callable that extracts (map_scores, overall_score) from a raw chunk result.

        :return: Extractor callable.
        """


class DSPySentimentAnalysis(SentimentAnalysisBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    """DSPy bridge for sentiment analysis."""

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
    def _prompt_conclusion(self) -> str | None:
        return None

    @override
    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return self._model_settings.inference_mode or dspy_.InferenceMode.predict

    @override
    def _get_extractor(self) -> Callable[[Any], tuple[dict[str, float], float | None]]:
        def extractor(res: Any) -> tuple[dict[str, float], float | None]:
            m_scores = {aspect: float(getattr(res, aspect)) for aspect in self._aspects}
            return m_scores, getattr(res, "score", None)

        return extractor

    @override
    def integrate(self, results: Sequence[dspy_.Result], docs: list[Doc]) -> list[Doc]:
        for doc, result in zip(docs, results):
            # Take the first completion.
            prediction = result.completions[0]
            label_scores = {aspect: float(getattr(prediction, aspect)) for aspect in self._aspects}
            doc.results[self._task_id] = Result(
                sentiment_per_aspect=label_scores,
                score=getattr(prediction, "score", None),
            )
        return docs

    @override
    def consolidate(
        self, results: Sequence[dspy_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[dspy_.Result]:
        consolidated_results_clean = self._consolidation_strategy.consolidate(results, docs_offsets)

        # Wrap back into dspy.Prediction.
        consolidated_results: list[dspy_.Result] = []
        for aspect_scores, overall_score in consolidated_results_clean:
            data = {**aspect_scores, "score": overall_score}
            consolidated_results.append(
                dspy.Prediction.from_completions(
                    [data],
                    signature=self.prompt_signature,
                )
            )
        return consolidated_results


class PydanticBasedSentAnalysis(
    SentimentAnalysisBridge[pydantic.BaseModel, pydantic.BaseModel, ModelWrapperInferenceMode], abc.ABC
):
    """Base class for Pydantic-based sentiment analysis bridges."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return (
            f"""
        Perform aspect-based sentiment analysis of the provided text given the provided aspects:
        {",".join(self._aspects)}."""
            + """
        For each aspect, provide the sentiment in the provided text with respect to this aspect.
        The "overall" aspect should reflect the sentiment in the text overall.
        A score of 1.0 means that the sentiment in the text with respect to this aspect is extremely positive.
        0 means the opposite, 0.5 means neutral.
        The sentiment score per aspect should ALWAYS be between 0 and 1.
        Also provide an overall confidence score between 0.0 and 1.0 for the sentiment analysis.

        The output for two aspects ASPECT_1 and ASPECT_2 should look like this:
        <output>
            <aspect_sentiments>
                <aspect_sentiment>
                    <aspect>ASPECT_1</aspect>
                    <sentiment>SENTIMENT_SCORE_1</sentiment>
                <aspect_sentiment>
                <aspect_sentiment>
                    <aspect>ASPECT_2</aspect>
                    <sentiment>SENTIMENT_SCORE_2</sentiment>
                <aspect_sentiment>
            </aspect_sentiments>
            <score>CONFIDENCE_SCORE</score>
        </output>
        """
        )

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
                        <aspect_sentiments>
                        {%- for a, s in example.sentiment_per_aspect.items() %}
                            <aspect_sentiment>
                                <aspect>{{ a }}</aspect>
                                <sentiment>{{ s }}</sentiment>
                            </aspect_sentiment>
                        {% endfor -%}
                        </aspect_sentiments>
                        <score>{{ example.score }}</score>
                    </output>
                </example>
            {% endfor %}
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
    def _get_extractor(self) -> Callable[[Any], tuple[dict[str, float], float | None]]:
        def extractor(res: Any) -> tuple[dict[str, float], float | None]:
            m_scores = {aspect: float(getattr(res, aspect)) for aspect in self._aspects}
            return m_scores, getattr(res, "score", None)

        return extractor

    @override
    def integrate(self, results: Sequence[pydantic.BaseModel], docs: list[Doc]) -> list[Doc]:
        for doc, result in zip(docs, results):
            label_scores = {k: v for k, v in result.model_dump().items() if k not in ["score"]}
            doc.results[self._task_id] = Result(sentiment_per_aspect=label_scores, score=getattr(result, "score", None))
        return docs

    @override
    def consolidate(
        self, results: Sequence[pydantic.BaseModel], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[pydantic.BaseModel]:
        assert issubclass(self.prompt_signature, pydantic.BaseModel)

        consolidated_results_clean = self._consolidation_strategy.consolidate(results, docs_offsets)
        consolidated_results: list[pydantic.BaseModel] = []

        for aspect_scores, overall_score in consolidated_results_clean:
            consolidated_results.append(
                self.prompt_signature(
                    **aspect_scores,
                    score=overall_score,
                )
            )

        return consolidated_results


class OutlinesSentimentAnalysis(PydanticBasedSentAnalysis[outlines_.InferenceMode]):
    """Outlines bridge for sentiment analysis."""

    @override
    @property
    def model_type(self) -> ModelType:
        return ModelType.outlines

    @override
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return self._model_settings.inference_mode or outlines_.InferenceMode.json


class LangChainSentimentAnalysis(PydanticBasedSentAnalysis[langchain_.InferenceMode]):
    """LangChain bridge for sentiment analysis."""

    @override
    @property
    def model_type(self) -> ModelType:
        return ModelType.langchain

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return super()._default_prompt_instructions

    @override
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return self._model_settings.inference_mode or langchain_.InferenceMode.structured
