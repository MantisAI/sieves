import abc
import warnings
from collections.abc import Iterable
from functools import cached_property
from typing import Any, Literal, TypeVar

import dspy
import jinja2
import pydantic

from sieves.data import Doc
from sieves.engines import dspy_, glix_, huggingface_, ollama_, outlines_
from sieves.tasks.core import Bridge

BridgePromptSignature = TypeVar("BridgePromptSignature", covariant=True)
BridgeInferenceMode = TypeVar("BridgeInferenceMode", covariant=True)
BridgeResult = TypeVar("BridgeResult")


class ClassificationBridge(
    Bridge[BridgePromptSignature, BridgeInferenceMode, BridgeResult],
    abc.ABC,
):
    def __init__(self, task_id: str, custom_prompt_template: str | None, labels: list[str]):
        """
        Initializes InformationExtractionBridge.
        :param task_id: Task ID.
        :param custom_prompt_template: Custom prompt template.
        :param labels: Labels to classify.
        """
        super().__init__(task_id, custom_prompt_template)
        self._labels = labels


class DSPyClassification(ClassificationBridge[dspy_.PromptSignature, dspy_.InferenceMode, dspy_.Result]):
    @property
    def prompt_template(self) -> str | None:
        return (
            self._custom_prompt_template
            or """
            Perform multi-label classification of the provided text given the provided labels.
            For each label, provide the conficence with which you believe that the provided text should be assigned 
            this label. A confidence of 1.0 means that this text should absolutely be assigned this label. 0 means the 
            opposite. Confidence per label should always be between 0 and 1. Confidence across lables does not have to 
            add up to 1.
            """
        )

    @cached_property
    def prompt_signature(self) -> type[dspy_.PromptSignature]:  # type: ignore[valid-type]
        labels = self._labels
        # Dynamically create Literal as output type.
        LabelType = Literal[*labels]  # type: ignore[valid-type]

        class TextClassification(dspy.Signature):  # type: ignore[misc]
            text: str = dspy.InputField()
            confidence_per_label: dict[LabelType, float] = dspy.OutputField()

        TextClassification.__doc__ = jinja2.Template(self.prompt_template).render()

        return TextClassification

    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return dspy_.InferenceMode.predict

    def extract(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        return ({"text": doc.text if doc.text else None} for doc in docs)

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
            for res in results[doc_offset[0] : doc_offset[1]]:
                assert len(res.completions.confidence_per_label) == 1
                for label, score in res.completions.confidence_per_label[0].items():
                    # Clamp label to range between 0 and 1. Alternatively we could force this in the prompt signature,
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
                {"confidence_per_label": [{sls["label"]: sls["score"] for sls in sorted_label_scores}]},
                signature=self.prompt_signature,
            )


class HuggingFaceClassification(ClassificationBridge[list[str], huggingface_.InferenceMode, huggingface_.Result]):
    @property
    def prompt_template(self) -> str | None:
        return (
            self._custom_prompt_template
            or """
        This text is about {}.
        {% if examples|length > 0 -%}
            Examples:
        ----------
        {%- for example in examples %}
        Text: "{{ example.text }}":
        Output: 
        {% for l, s in example.confidence_per_label.items() %}    {{ l }}: {{ s }},
        {% endfor -%}
        {% endfor -%}
        ----------
        {% endif -%}
        """
        )

    @property
    def prompt_signature(self) -> list[str]:
        return self._labels

    @property
    def inference_mode(self) -> huggingface_.InferenceMode:
        return huggingface_.InferenceMode.zeroshot_cls

    def extract(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        return ({"text": doc.text if doc.text else None} for doc in docs)

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


GliXResult = list[dict[str, str | float]]


class GliXClassification(ClassificationBridge[list[str], glix_.InferenceMode, GliXResult]):
    def __init__(self, task_id: str, custom_prompt_template: str | None, labels: list[str]):
        super().__init__(task_id=task_id, labels=labels, custom_prompt_template=custom_prompt_template)
        if self._custom_prompt_template:
            warnings.warn("Custom prompt template is ignored by GliX engine.")

    @property
    def prompt_template(self) -> str | None:
        return None

    @property
    def prompt_signature(self) -> list[str]:
        return self._labels

    @property
    def inference_mode(self) -> glix_.InferenceMode:
        return glix_.InferenceMode.gliclass

    def extract(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        return ({"text": doc.text if doc.text else None} for doc in docs)

    def integrate(self, results: Iterable[GliXResult], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            doc.results[self._task_id] = [
                (res["label"], res["score"]) for res in sorted(result, key=lambda x: x["score"], reverse=True)
            ]
        return docs

    def consolidate(self, results: Iterable[GliXResult], docs_offsets: list[tuple[int, int]]) -> Iterable[GliXResult]:
        results = list(results)

        # Determine label scores for chunks per document.
        for doc_offset in docs_offsets:
            label_scores: dict[str, float] = {label: 0.0 for label in self._labels}

            for rec in results[doc_offset[0] : doc_offset[1]]:
                for entry in rec:
                    assert isinstance(entry["label"], str)
                    assert isinstance(entry["score"], float)
                    label_scores[entry["label"]] += entry["score"]

            # Average score, sort by it in descending order.
            sorted_label_scores: list[dict[str, str | float]] = sorted(
                [
                    {"label": label, "score": score / (doc_offset[1] - doc_offset[0])}
                    for label, score in label_scores.items()
                ],
                key=lambda x: x["score"],
                reverse=True,
            )
            yield sorted_label_scores


class OutlinesClassification(
    ClassificationBridge[type[pydantic.BaseModel], outlines_.InferenceMode, pydantic.BaseModel]
):
    @property
    def prompt_template(self) -> str | None:
        return (
            self._custom_prompt_template
            or f"""
            Perform multi-label classification of the provided text given the provided labels: {",".join(self._labels)}.
            For each label, provide the conficence with which you believe that the provided text should be assigned
            this label. A confidence of 1.0 means that this text should absolutely be assigned this label. 0 means the
            opposite. Confidence per label should ALWAYS be between 0 and 1.

            The output for two labels LABEL_1 and LABEL_2 should look like this:
            Output: {{
                LABEL_1: CONFIDENCE_SCORE_1,
                LABEL_2: CONFIDENCE_SCORE_2,
            }}

            {{% if examples|length > 0 -%}}
                Examples:
                ----------
                {{%- for example in examples %}}
                    Text: "{{{{ example.text }}}}":
                    Output: 
                    {{% for l, s in example.confidence_per_label.items() %}}    {{{{ l }}}}: {{{{ s }}}},
                    {{% endfor -%}}
                {{% endfor %}}
                ----------
            {{% endif -%}}

            ========
            Text: {{{{ text }}}}
            Output: 
            """
        )

    @cached_property
    def prompt_signature(self) -> type[pydantic.BaseModel]:
        prompt_sig = pydantic.create_model(  # type: ignore[call-overload]
            "MultilabelPrediction",
            __base__=pydantic.BaseModel,
            **{label: (float, ...) for label in self._labels},
        )

        assert isinstance(prompt_sig, type) and issubclass(prompt_sig, pydantic.BaseModel)
        return prompt_sig

    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return outlines_.InferenceMode.json

    def extract(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        return ({"text": doc.text if doc.text else None} for doc in docs)

    def integrate(self, results: Iterable[pydantic.BaseModel], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            doc.results[self._task_id] = sorted(
                [(label, score) for label, score in result.model_dump().items()], key=lambda x: x[1], reverse=True
            )
        return docs

    def consolidate(
        self, results: Iterable[pydantic.BaseModel], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[pydantic.BaseModel]:
        results = list(results)

        # Determine label scores for chunks per document.
        for doc_offset in docs_offsets:
            label_scores: dict[str, float] = {label: 0.0 for label in self._labels}
            for rec in results[doc_offset[0] : doc_offset[1]]:
                for label in self._labels:
                    # Clamp label to range between 0 and 1. Alternatively we could force this in the prompt signature,
                    # but this fails occasionally with some models and feels too strict (maybe a strict mode would be
                    # useful?).
                    label_scores[label] += max(0, min(getattr(rec, label), 1))

            yield self.prompt_signature(
                **{label: score / (doc_offset[1] - doc_offset[0]) for label, score in label_scores.items()}
            )


class OllamaClassification(ClassificationBridge[type[pydantic.BaseModel], ollama_.InferenceMode, pydantic.BaseModel]):
    @property
    def prompt_template(self) -> str | None:
        return (
            self._custom_prompt_template
            or f"""
            Perform multi-label classification of the provided text given the provided labels: {",".join(self._labels)}.
            For each label, provide the conficence with which you believe that the provided text should be assigned
            this label. A confidence of 1.0 means that this text should absolutely be assigned this label. 0 means the
            opposite. Confidence per label should ALWAYS be between 0 and 1.

            The output for two labels LABEL_1 and LABEL_2 should look like this:
            Output:
                LABEL_1: CONFIDENCE_SCORE_1,
                LABEL_2: CONFIDENCE_SCORE_2,

            {{% if examples|length > 0 -%}}
                Examples:
                ----------
                {{%- for example in examples %}}
                    Text: "{{{{ example.text }}}}":
                    Output: 
                    {{% for l, s in example.confidence_per_label.items() %}}    {{{{ l }}}}: {{{{ s }}}},
                    {{% endfor -%}}
                {{% endfor %}}
                ----------
            {{% endif -%}}

            ========
            Text: {{{{ text }}}}
            Output: 
            """
        )

    @cached_property
    def prompt_signature(self) -> type[pydantic.BaseModel]:
        prompt_sig = pydantic.create_model(  # type: ignore[call-overload]
            "MultilabelPrediction",
            __base__=pydantic.BaseModel,
            **{label: (float, ...) for label in self._labels},
        )

        assert isinstance(prompt_sig, type) and issubclass(prompt_sig, pydantic.BaseModel)
        return prompt_sig

    @property
    def inference_mode(self) -> ollama_.InferenceMode:
        return ollama_.InferenceMode.chat

    def extract(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        return ({"text": doc.text if doc.text else None} for doc in docs)

    def integrate(self, results: Iterable[pydantic.BaseModel], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            doc.results[self._task_id] = sorted(
                [(label, score) for label, score in result.model_dump().items()], key=lambda x: x[1], reverse=True
            )
        return docs

    def consolidate(
        self, results: Iterable[pydantic.BaseModel], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[pydantic.BaseModel]:
        results = list(results)

        # Determine label scores for chunks per document.
        for doc_offset in docs_offsets:
            label_scores: dict[str, float] = {label: 0.0 for label in self._labels}
            for rec in results[doc_offset[0] : doc_offset[1]]:
                for label in self._labels:
                    # Clamp label to range between 0 and 1. Alternatively we could force this in the prompt signature,
                    # but this fails occasionally with some models and feels too strict (maybe a strict mode would be
                    # useful?).
                    label_scores[label] += max(0, min(getattr(rec, label), 1))

            yield self.prompt_signature(
                **{label: score / (doc_offset[1] - doc_offset[0]) for label, score in label_scores.items()}
            )
