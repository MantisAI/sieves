import abc
from collections.abc import Iterable
from functools import cached_property
from typing import Literal, TypeVar

import dspy
import jinja2
import pydantic

from sieves.data import Doc
from sieves.engines import EngineInferenceMode, dspy_, instructor_, langchain_, ollama_, outlines_, vllm_
from sieves.tasks.predictive.bridges import Bridge

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class SentAnalysisBridge(Bridge[_BridgePromptSignature, _BridgeResult, EngineInferenceMode], abc.ABC):
    def __init__(
        self, task_id: str, prompt_template: str | None, prompt_signature_desc: str | None, aspects: tuple[str, ...]
    ):
        """
        Initializes SentAnalysisBridge.
        :param task_id: Task ID.
        :param prompt_template: Custom prompt template.
        :param prompt_signature_desc: Custom prompt signature description.
        :param aspects: Aspects to consider.
        """
        super().__init__(
            task_id=task_id,
            prompt_template=prompt_template,
            prompt_signature_desc=prompt_signature_desc,
            overwrite=False,
        )
        self._aspects = aspects


class DSPySentimentAnalysis(SentAnalysisBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    @property
    def _prompt_template(self) -> str | None:
        return None

    @property
    def _prompt_signature_description(self) -> str | None:
        return """
        Aspect-based sentiment analysis of the provided text given the provided aspects.
        For each aspect, provide the sentiment score with which you reflects the sentiment in the provided text with 
        respect to this aspect.
        The "overall" aspect should reflect the sentiment in the text overall. 
        A score of 1.0 means that the sentiment in the text with respect to this aspect is extremely positive. 
        0 means the opposite, 0.5 means neutral. 
        Sentiment per aspect should always be between 0 and 1. 
        """

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

        SentimentAnalysis.__doc__ = jinja2.Template(self.prompt_signature_description).render()

        return SentimentAnalysis

    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return dspy_.InferenceMode.chain_of_thought

    def integrate(self, results: Iterable[dspy_.Result], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            assert len(result.completions.sentiment_per_aspect) == 1
            sorted_preds = sorted(
                [(aspect, score) for aspect, score in result.completions.sentiment_per_aspect[0].items()],
                key=lambda x: x[1],
                reverse=True,
            )
            doc.results[self._task_id] = sorted_preds
        return docs

    def consolidate(
        self, results: Iterable[dspy_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[dspy_.Result]:
        results = list(results)

        # Determine label scores for chunks per document.
        for doc_offset in docs_offsets:
            aspect_scores: dict[str, float] = {label: 0.0 for label in self._aspects}
            doc_results = results[doc_offset[0] : doc_offset[1]]

            for res in doc_results:
                assert len(res.completions.sentiment_per_aspect) == 1
                for label, score in res.completions.sentiment_per_aspect[0].items():
                    # Clamp score to range between 0 and 1. Alternatively we could force this in the prompt signature,
                    # but this fails occasionally with some models and feels too strict (maybe a strict mode would be
                    # useful?).
                    aspect_scores[label] += max(0, min(score, 1))

            sorted_aspect_scores: list[dict[str, str | float]] = sorted(
                [
                    {"aspect": aspect, "score": score / (doc_offset[1] - doc_offset[0])}
                    for aspect, score in aspect_scores.items()
                ],
                key=lambda x: x["score"],
                reverse=True,
            )

            yield dspy.Prediction.from_completions(
                {
                    "sentiment_per_aspect": [{sls["aspect"]: sls["score"] for sls in sorted_aspect_scores}],
                    "reasoning": [str([res.reasoning for res in doc_results])],
                },
                signature=self.prompt_signature,
            )


class PydanticBasedSentAnalysis(
    SentAnalysisBridge[pydantic.BaseModel, pydantic.BaseModel, EngineInferenceMode], abc.ABC
):
    @property
    def _prompt_template(self) -> str | None:
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

        ========
        
        <text>{{ text }}</text>
        <output>
        """
        )

    @property
    def _prompt_signature_description(self) -> str | None:
        return None

    @cached_property
    def prompt_signature(self) -> type[pydantic.BaseModel]:
        prompt_sig = pydantic.create_model(  # type: ignore[call-overload]
            "SentimentAnalysis",
            __base__=pydantic.BaseModel,
            reasoning=(str, ...),
            **{aspect: (float, ...) for aspect in self._aspects},
        )

        if self.prompt_signature_description:
            prompt_sig.__doc__ = jinja2.Template(self.prompt_signature_description).render()

        assert isinstance(prompt_sig, type) and issubclass(prompt_sig, pydantic.BaseModel)
        return prompt_sig

    def integrate(self, results: Iterable[pydantic.BaseModel], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            label_scores = {k: v for k, v in result.model_dump().items() if k != "reasoning"}
            doc.results[self._task_id] = sorted(
                [(aspect, score) for aspect, score in label_scores.items()], key=lambda x: x[1], reverse=True
            )
        return docs

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
                reasonings.append(rec.reasoning)
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
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return outlines_.InferenceMode.json


class OllamaSentimentAnalysis(PydanticBasedSentAnalysis[ollama_.InferenceMode]):
    @property
    def inference_mode(self) -> ollama_.InferenceMode:
        return ollama_.InferenceMode.structured


class LangChainSentimentAnalysis(PydanticBasedSentAnalysis[langchain_.InferenceMode]):
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return langchain_.InferenceMode.structured


class InstructorSentimentAnalysis(PydanticBasedSentAnalysis[instructor_.InferenceMode]):
    @property
    def inference_mode(self) -> instructor_.InferenceMode:
        return instructor_.InferenceMode.structured


class VLLMSentimentAnalysis(PydanticBasedSentAnalysis[vllm_.InferenceMode]):
    @property
    def inference_mode(self) -> vllm_.InferenceMode:
        return vllm_.InferenceMode.json
