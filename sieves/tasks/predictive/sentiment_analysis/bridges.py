"""Bridges for sentiment analysis task."""

import abc
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any, Literal, TypeVar, override

import dspy
import jinja2
import pydantic

from sieves.data import Doc
from sieves.model_wrappers import ModelWrapperInferenceMode, dspy_, langchain_, outlines_
from sieves.model_wrappers.types import ModelSettings
from sieves.tasks.predictive.bridges import Bridge
from sieves.tasks.predictive.consolidation import MapScoreConsolidation
from sieves.tasks.predictive.schemas.sentiment_analysis import Result

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class SentAnalysisBridge(Bridge[_BridgePromptSignature, _BridgeResult, ModelWrapperInferenceMode], abc.ABC):
    """Abstract base class for sentiment analysis bridges."""

    def __init__(
        self,
        task_id: str,
        prompt_instructions: str | None,
        aspects: tuple[str, ...],
        model_settings: ModelSettings,
    ):
        """Initialize SentAnalysisBridge.

        :param task_id: Task ID.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param aspects: Aspects to consider.
        :param model_settings: Model settings including inference_mode.
        """
        super().__init__(
            task_id=task_id,
            prompt_instructions=prompt_instructions,
            overwrite=False,
            model_settings=model_settings,
        )
        self._aspects = aspects
        self._consolidation_strategy = MapScoreConsolidation(
            keys=self._aspects,
            extractor=self._get_extractor(),
        )

    @abc.abstractmethod
    def _get_extractor(self) -> Callable[[Any], tuple[dict[str, float], float | None]]:
        """Return a callable that extracts (map_scores, overall_score) from a raw chunk result.

        :return: Extractor callable.
        """


class DSPySentimentAnalysis(SentAnalysisBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    """DSPy bridge for sentiment analysis."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return """
        Aspect-based sentiment analysis of the provided text given the provided aspects.
        For each aspect, provide the sentiment score with which you reflects the sentiment in the provided text with
        respect to this aspect.
        The "overall" aspect should reflect the sentiment in the text overall.
        A score of 1.0 means that the sentiment in the text with respect to this aspect is extremely positive.
        0 means the opposite, 0.5 means neutral.
        Sentiment per aspect should always be between 0 and 1.
        Also provide an overall confidence score between 0.0 and 1.0 for the sentiment analysis.
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
        aspects = self._aspects
        # Dynamically create Literal as output type.
        AspectType = Literal[*aspects]  # type: ignore[valid-type]

        class SentimentAnalysis(dspy.Signature):  # type: ignore[misc]
            text: str = dspy.InputField(description="Text to determine sentiments for.")
            sentiment_per_aspect: dict[AspectType, float] = dspy.OutputField(
                description="Sentiment in this text with respect to the corresponding aspect."
            )
            score: float = dspy.OutputField(description="Overall confidence score between 0.0 and 1.0.")

        SentimentAnalysis.__doc__ = jinja2.Template(self._prompt_instructions).render()

        return SentimentAnalysis

    @override
    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return self._model_settings.inference_mode or dspy_.InferenceMode.predict

    @override
    def _get_extractor(self) -> Callable[[Any], tuple[dict[str, float], float | None]]:
        def extractor(res: Any) -> tuple[dict[str, float], float | None]:
            return res.sentiment_per_aspect, getattr(res, "score", None)

        return extractor

    @override
    def integrate(self, results: Sequence[dspy_.Result], docs: list[Doc]) -> list[Doc]:
        for doc, result in zip(docs, results):
            assert len(result.completions.sentiment_per_aspect) == 1
            doc.results[self._task_id] = Result(
                sentiment_per_aspect=result.completions.sentiment_per_aspect[0],
                score=getattr(result, "score", None),
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
            # Sort by score descending as in original implementation.
            sorted_aspect_scores = sorted(
                aspect_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            consolidated_results.append(
                dspy.Prediction.from_completions(
                    {
                        "sentiment_per_aspect": [{k: v for k, v in sorted_aspect_scores}],
                        "score": [overall_score],
                    },
                    signature=self.prompt_signature,
                )
            )
        return consolidated_results


class PydanticBasedSentAnalysis(
    SentAnalysisBridge[pydantic.BaseModel, pydantic.BaseModel, ModelWrapperInferenceMode], abc.ABC
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
    @cached_property
    def prompt_signature(self) -> type[pydantic.BaseModel]:
        fields = {aspect: (float, ...) for aspect in self._aspects}
        fields["score"] = (float | None, None)
        prompt_sig = pydantic.create_model(  # type: ignore[no-matching-overload]
            "SentimentAnalysis",
            __base__=pydantic.BaseModel,
            __doc__="Sentiment analysis of specified text.",
            **fields,
        )

        assert isinstance(prompt_sig, type) and issubclass(prompt_sig, pydantic.BaseModel)
        return prompt_sig

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
    def inference_mode(self) -> outlines_.InferenceMode:
        return self._model_settings.inference_mode or outlines_.InferenceMode.json


class LangChainSentimentAnalysis(PydanticBasedSentAnalysis[langchain_.InferenceMode]):
    """LangChain bridge for sentiment analysis."""

    @override
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return self._model_settings.inference_mode or langchain_.InferenceMode.structured
