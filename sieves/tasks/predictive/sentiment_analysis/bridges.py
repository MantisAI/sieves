"""Bridges for sentiment analysis task."""

import abc
from collections.abc import Iterable
from functools import cached_property
from typing import Literal, TypeVar, override

import dspy
import jinja2
import pydantic

from sieves.data import Doc
from sieves.engines import EngineInferenceMode, dspy_, langchain_, outlines_
from sieves.engines.types import GenerationSettings
from sieves.tasks.predictive.bridges import Bridge

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class SentAnalysisBridge(Bridge[_BridgePromptSignature, _BridgeResult, EngineInferenceMode], abc.ABC):
    """Abstract base class for sentiment analysis bridges."""

    def __init__(
        self,
        task_id: str,
        prompt_instructions: str | None,
        aspects: tuple[str, ...],
        generation_settings: GenerationSettings,
    ):
        """Initialize SentAnalysisBridge.

        :param task_id: Task ID.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param aspects: Aspects to consider.
        :param generation_settings: Generation settings including inference_mode.
        """
        super().__init__(
            task_id=task_id,
            prompt_instructions=prompt_instructions,
            overwrite=False,
            generation_settings=generation_settings,
        )
        self._aspects = aspects


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
            reasoning: str | None = dspy.OutputField(
                default=None, description="Provide reasoning for aspect-based sentiment assessments when beneficial."
            )
            sentiment_per_aspect: dict[AspectType, float] = dspy.OutputField(
                description="Sentiment in this text with respect to the corresponding aspect."
            )

        SentimentAnalysis.__doc__ = jinja2.Template(self._prompt_instructions).render()

        return SentimentAnalysis

    @override
    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return self._generation_settings.inference_mode or dspy_.InferenceMode.predict

    @override
    def integrate(self, results: Iterable[dspy_.Result], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            assert len(result.completions.sentiment_per_aspect) == 1
            sorted_preds = sorted(
                ((aspect, score) for aspect, score in result.completions.sentiment_per_aspect[0].items()),
                key=lambda x: x[1],
                reverse=True,
            )
            doc.results[self._task_id] = sorted_preds
        return docs

    @override
    def consolidate(
        self, results: Iterable[dspy_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[dspy_.Result]:
        results = list(results)

        # Determine label scores for chunks per document.
        for doc_offset in docs_offsets:
            aspect_scores: dict[str, float] = {label: 0.0 for label in self._aspects}
            doc_results = results[doc_offset[0] : doc_offset[1]]

            for res in doc_results:
                if res is None:
                    continue

                assert len(res.completions.sentiment_per_aspect) == 1
                for label, score in res.completions.sentiment_per_aspect[0].items():
                    # Clamp score to range between 0 and 1. Alternatively we could force this in the prompt signature,
                    # but this fails occasionally with some models and feels too strict (maybe a strict mode would be
                    # useful?).
                    aspect_scores[label] += max(0, min(score, 1))

            sorted_aspect_scores: list[dict[str, str | float]] = sorted(
                (
                    {"aspect": aspect, "score": score / (doc_offset[1] - doc_offset[0])}
                    for aspect, score in aspect_scores.items()
                ),
                key=lambda x: x["score"],
                reverse=True,
            )

            yield dspy.Prediction.from_completions(
                {
                    "sentiment_per_aspect": [{sls["aspect"]: sls["score"] for sls in sorted_aspect_scores}],
                    "reasoning": [str([getattr(res, "reasoning", "") for res in doc_results if res is not None])],
                },
                signature=self.prompt_signature,
            )


class PydanticBasedSentAnalysis(
    SentAnalysisBridge[pydantic.BaseModel, pydantic.BaseModel, EngineInferenceMode], abc.ABC
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
        The sentiment score per aspect should ALWAYS be between 0 and 1. Provide the reasoning for your decision.

        The output for two aspects ASPECT_1 and ASPECT_2 should look like this:
        <output>
            <reasoning>REASONING</reasoning>
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
                        <reasoning>{{ example.reasoning }}</reasoning>
                        <aspect_sentiments>
                        {%- for a, s in example.sentiment_per_aspect.items() %}
                            <aspect_sentiment>
                                <aspect>{{ a }}</aspect>
                                <sentiment>{{ s }}</sentiment>
                            </aspect_sentiment>
                        {% endfor -%}
                        </aspect_sentiments>
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
        prompt_sig = pydantic.create_model(  # type: ignore[no-matching-overload]
            "SentimentAnalysis",
            __base__=pydantic.BaseModel,
            __doc__="Sentiment analysis of specified text.",
            reasoning=(
                str | None,
                pydantic.Field(
                    default=None,
                    description="Provide reasoning for aspect-based sentiment assessments when beneficial.",
                ),
            ),
            **{aspect: (float, ...) for aspect in self._aspects},
        )

        assert isinstance(prompt_sig, type) and issubclass(prompt_sig, pydantic.BaseModel)
        return prompt_sig

    @override
    def integrate(self, results: Iterable[pydantic.BaseModel], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            label_scores = {k: v for k, v in result.model_dump().items() if k != "reasoning"}
            doc.results[self._task_id] = sorted(
                ((aspect, score) for aspect, score in label_scores.items()), key=lambda x: x[1], reverse=True
            )
        return docs

    @override
    def consolidate(
        self, results: Iterable[pydantic.BaseModel], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[pydantic.BaseModel]:
        results = list(results)

        # Determine label scores for chunks per document.
        reasonings: list[str] = []
        for doc_offset in docs_offsets:
            aspect_scores: dict[str, float] = {label: 0.0 for label in self._aspects}
            doc_results = results[doc_offset[0] : doc_offset[1]]

            for rec in doc_results:
                if rec is None:
                    continue  # type: ignore[unreachable]

                assert hasattr(rec, "reasoning")
                reasonings.append(rec.reasoning or "")
                for aspect in self._aspects:
                    # Clamp score to range between 0 and 1. Alternatively we could force this in the prompt signature,
                    # but this fails occasionally with some models and feels too strict (maybe a strict mode would be
                    # useful?).
                    aspect_scores[aspect] += max(0, min(getattr(rec, aspect), 1))

            yield self.prompt_signature(
                reasoning=str(reasonings),
                **{aspect: score / (doc_offset[1] - doc_offset[0]) for aspect, score in aspect_scores.items()},
            )


class OutlinesSentimentAnalysis(PydanticBasedSentAnalysis[outlines_.InferenceMode]):
    """Outlines bridge for sentiment analysis."""

    @override
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return self._generation_settings.inference_mode or outlines_.InferenceMode.json


class LangChainSentimentAnalysis(PydanticBasedSentAnalysis[langchain_.InferenceMode]):
    """LangChain bridge for sentiment analysis."""

    @override
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return self._generation_settings.inference_mode or langchain_.InferenceMode.structured
