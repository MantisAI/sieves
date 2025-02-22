import abc
from collections.abc import Iterable
from functools import cached_property
from typing import Literal, TypeVar

import dspy
import jinja2
import pydantic

from sieves.data import Doc
from sieves.engines import EngineInferenceMode, dspy_, huggingface_, instructor_, langchain_, ollama_, outlines_
from sieves.tasks.predictive.bridges import Bridge

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class ClassificationBridge(Bridge[_BridgePromptSignature, _BridgeResult, EngineInferenceMode], abc.ABC):
    def __init__(self, task_id: str, prompt_template: str | None, prompt_signature_desc: str | None, labels: list[str]):
        """
        Initializes InformationExtractionBridge.
        :param task_id: Task ID.
        :param prompt_template: Custom prompt template.
        :param prompt_signature_desc: Custom prompt signature description.
        :param labels: Labels to classify.
        """
        super().__init__(task_id=task_id, prompt_template=prompt_template, prompt_signature_desc=prompt_signature_desc)
        self._labels = labels


class DSPyClassification(ClassificationBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    @property
    def _prompt_template(self) -> str | None:
        return None

    @property
    def _prompt_signature_description(self) -> str | None:
        return """
        Multi-label classification of the provided text given the provided labels.
        For each label, provide the confidence with which you believe that the provided text should be assigned 
        this label. A confidence of 1.0 means that this text should absolutely be assigned this label. 0 means the 
        opposite. Confidence per label should always be between 0 and 1. Confidence across lables does not have to 
        add up to 1.
        """

    @cached_property
    def prompt_signature(self) -> type[dspy_.PromptSignature]:
        labels = self._labels
        # Dynamically create Literal as output type.
        LabelType = Literal[*labels]  # type: ignore[valid-type]

        class TextClassification(dspy.Signature):  # type: ignore[misc]
            text: str = dspy.InputField(description="Text to classify.")
            confidence_per_label: dict[LabelType, float] = dspy.OutputField(
                description="Confidence per label that text should be classified with this label."
            )

        TextClassification.__doc__ = jinja2.Template(self.prompt_signature_description).render()

        return TextClassification

    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return dspy_.InferenceMode.chain_of_thought

    def integrate(self, results: Iterable[dspy_.Result], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            assert len(result.completions.confidence_per_label) == 1
            sorted_preds = sorted(
                [(label, score) for label, score in result.completions.confidence_per_label[0].items()],
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
            label_scores: dict[str, float] = {label: 0.0 for label in self._labels}
            doc_results = results[doc_offset[0] : doc_offset[1]]

            for res in doc_results:
                assert len(res.completions.confidence_per_label) == 1
                for label, score in res.completions.confidence_per_label[0].items():
                    # Clamp score to range between 0 and 1. Alternatively we could force this in the prompt signature,
                    # but this fails occasionally with some models and feels too strict (maybe a strict mode would be
                    # useful?).
                    label_scores[label] += max(0, min(score, 1))

            sorted_label_scores: list[dict[str, str | float]] = sorted(
                [
                    {"label": label, "score": score / (doc_offset[1] - doc_offset[0])}
                    for label, score in label_scores.items()
                ],
                key=lambda x: x["score"],
                reverse=True,
            )

            yield dspy.Prediction.from_completions(
                {
                    "confidence_per_label": [{sls["label"]: sls["score"] for sls in sorted_label_scores}],
                    "reasoning": [str([res.reasoning for res in doc_results])],
                },
                signature=self.prompt_signature,
            )


class HuggingFaceClassification(ClassificationBridge[list[str], huggingface_.Result, huggingface_.InferenceMode]):
    @property
    def _prompt_template(self) -> str | None:
        return """
        This text is about {}.
        {% if examples|length > 0 -%}
            Examples:
        ----------
        {%- for example in examples %}
        Text: "{{ example.text }}"
        Reasoning: "{{ example.reasoning }}"
        Output: 
        {% for l, s in example.confidence_per_label.items() %}    {{ l }}: {{ s }},
        {% endfor -%}
        {% endfor -%}
        ----------
        {% endif -%}
        """

    @property
    def _prompt_signature_description(self) -> str | None:
        return None

    @property
    def prompt_signature(self) -> list[str]:
        return self._labels

    @property
    def inference_mode(self) -> huggingface_.InferenceMode:
        return huggingface_.InferenceMode.zeroshot_cls

    def integrate(self, results: Iterable[huggingface_.Result], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            doc.results[self._task_id] = [(label, score) for label, score in zip(result["labels"], result["scores"])]
        return docs

    def consolidate(
        self, results: Iterable[huggingface_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[huggingface_.Result]:
        results = list(results)

        # Determine label scores for chunks per document.
        for doc_offset in docs_offsets:
            label_scores: dict[str, float] = {label: 0.0 for label in self._labels}

            for rec in results[doc_offset[0] : doc_offset[1]]:
                for label, score in zip(rec["labels"], rec["scores"]):
                    assert isinstance(label, str)
                    assert isinstance(score, float)
                    label_scores[label] += score

            # Average score, sort by it in descending order.
            sorted_label_scores: list[dict[str, str | float]] = sorted(
                [
                    {"label": label, "score": score / (doc_offset[1] - doc_offset[0])}
                    for label, score in label_scores.items()
                ],
                key=lambda x: x["score"],
                reverse=True,
            )
            yield {
                "labels": [rec["label"] for rec in sorted_label_scores],  # type: ignore[dict-item]
                "scores": [rec["score"] for rec in sorted_label_scores],  # type: ignore[dict-item]
            }


class PydanticBasedClassification(
    ClassificationBridge[pydantic.BaseModel, pydantic.BaseModel, EngineInferenceMode], abc.ABC
):
    @property
    def _prompt_template(self) -> str | None:
        return f"""
        Perform multi-label classification of the provided text given the provided labels: {",".join(self._labels)}.
        For each label, provide the confidence with which you believe that the provided text should be assigned
        this label. A confidence of 1.0 means that this text should absolutely be assigned this label. 0 means the
        opposite. Confidence per label should ALWAYS be between 0 and 1. Provide the reasoning for your decision. 

        The output for two labels LABEL_1 and LABEL_2 should look like this:
        Output:
            Reasoning: REASONING
            LABEL_1: CONFIDENCE_SCORE_1
            LABEL_2: CONFIDENCE_SCORE_2

        {{% if examples|length > 0 -%}}
            Examples:
            ----------
            {{%- for example in examples %}}
                Text: "{{{{ example.text }}}}"
                Output:
                    Reasoning: "{{{{ example.reasoning }}}}" 
                {{% for l, s in example.confidence_per_label.items() %}}    {{{{ l }}}}: {{{{ s }}}},
                {{% endfor -%}}
            {{% endfor %}}
            ----------
        {{% endif -%}}

        ========
        Text: {{{{ text }}}}
        Output: 
        """

    @property
    def _prompt_signature_description(self) -> str | None:
        return None

    @cached_property
    def prompt_signature(self) -> type[pydantic.BaseModel]:
        prompt_sig = pydantic.create_model(  # type: ignore[call-overload]
            "MultilabelPrediction",
            __base__=pydantic.BaseModel,
            reasoning=(str, ...),
            **{label: (float, ...) for label in self._labels},
        )

        if self.prompt_signature_description:
            prompt_sig.__doc__ = jinja2.Template(self.prompt_signature_description).render()

        assert isinstance(prompt_sig, type) and issubclass(prompt_sig, pydantic.BaseModel)
        return prompt_sig

    def integrate(self, results: Iterable[pydantic.BaseModel], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            label_scores = {k: v for k, v in result.model_dump().items() if k != "reasoning"}
            doc.results[self._task_id] = sorted(
                [(label, score) for label, score in label_scores.items()], key=lambda x: x[1], reverse=True
            )
        return docs

    def consolidate(
        self, results: Iterable[pydantic.BaseModel], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[pydantic.BaseModel]:
        results = list(results)

        # Determine label scores for chunks per document.
        reasonings: list[str] = []
        for doc_offset in docs_offsets:
            label_scores: dict[str, float] = {label: 0.0 for label in self._labels}
            doc_results = results[doc_offset[0] : doc_offset[1]]

            for rec in doc_results:
                assert hasattr(rec, "reasoning")
                reasonings.append(rec.reasoning)
                for label in self._labels:
                    # Clamp score to range between 0 and 1. Alternatively we could force this in the prompt signature,
                    # but this fails occasionally with some models and feels too strict (maybe a strict mode would be
                    # useful?).
                    label_scores[label] += max(0, min(getattr(rec, label), 1))

            yield self.prompt_signature(
                reasoning=str(reasonings),
                **{label: score / (doc_offset[1] - doc_offset[0]) for label, score in label_scores.items()},
            )


class OutlinesClassification(PydanticBasedClassification[outlines_.InferenceMode]):
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return outlines_.InferenceMode.json


class OllamaClassification(PydanticBasedClassification[ollama_.InferenceMode]):
    @property
    def inference_mode(self) -> ollama_.InferenceMode:
        return ollama_.InferenceMode.chat


class LangChainClassification(PydanticBasedClassification[langchain_.InferenceMode]):
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return langchain_.InferenceMode.structured_output


class InstructorClassification(PydanticBasedClassification[instructor_.InferenceMode]):
    @property
    def inference_mode(self) -> instructor_.InferenceMode:
        return instructor_.InferenceMode.chat
