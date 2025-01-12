import abc
from collections.abc import Iterable
from functools import cached_property
from typing import Any, Generic, Literal, TypeAlias, TypeVar

import dspy

from sieves.data import Doc
from sieves.engines import dspy_, glix_, huggingface_, outlines_

TaskPromptSignature: TypeAlias = list[str] | type[dspy_.PromptSignature]  # type: ignore[valid-type]
TaskInferenceMode: TypeAlias = (
    outlines_.InferenceMode | dspy_.InferenceMode | huggingface_.InferenceMode | glix_.InferenceMode
)
TaskResult: TypeAlias = outlines_.Result | dspy_.Result | huggingface_.Result | glix_.Result

BridgePromptSignature = TypeVar("BridgePromptSignature", covariant=True)
BridgeInferenceMode = TypeVar("BridgeInferenceMode", covariant=True)
BridgeResult = TypeVar("BridgeResult")


class ClassificationBridge(abc.ABC, Generic[BridgePromptSignature, BridgeInferenceMode, BridgeResult]):
    def __init__(self, task_id: str, labels: list[str]):
        self._task_id = task_id
        self._labels = labels

    @property
    @abc.abstractmethod
    def prompt_template(self) -> str | None:
        ...

    @property
    @abc.abstractmethod
    def prompt_signature(self) -> BridgePromptSignature:
        ...

    @property
    @abc.abstractmethod
    def inference_mode(self) -> BridgeInferenceMode:
        ...

    @abc.abstractmethod
    def extract(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        ...

    @abc.abstractmethod
    def integrate(self, results: Iterable[BridgeResult], docs: Iterable[Doc]) -> Iterable[Doc]:
        ...

    @abc.abstractmethod
    def consolidate(
        self, results: Iterable[BridgeResult], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[BridgeResult]:
        ...


class DSPyClassification(ClassificationBridge[dspy_.PromptSignature, dspy_.InferenceMode, dspy_.Result]):
    @property
    def prompt_template(self) -> str | None:
        return None

    @cached_property
    def prompt_signature(self) -> type[dspy_.PromptSignature]:  # type: ignore[valid-type]
        labels = self._labels
        # Dynamically create Literal as output type.
        LabelType = Literal[*labels]  # type: ignore[valid-type]

        class TextClassification(dspy.Signature):  # type: ignore[misc]
            """Classify text as one of a set of labels. Include confidence of classification."""

            text: str = dspy.InputField()
            labels: LabelType = dspy.OutputField()
            confidence: float = dspy.OutputField()

        return TextClassification

    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return dspy_.InferenceMode.predict

    def extract(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        return ({"text": doc.text if doc.text else None} for doc in docs)

    def integrate(self, results: Iterable[dspy_.Result], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            doc.results[self._task_id] = result.completions.labels
        return docs

    def consolidate(
        self, results: Iterable[dspy_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[dspy_.Result]:
        results = list(results)

        # Determine label scores for chunks per document.
        for doc_offset in docs_offsets:
            label_scores: dict[str, float] = {label: 0.0 for label in self._labels}
            for res in results[doc_offset[0] : doc_offset[1]]:
                for label, score in zip(res.completions.labels, res.completions.confidence):
                    label_scores[label] += score

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
                    "labels": [sls["label"] for sls in sorted_label_scores],
                    "confidence": [sls["score"] for sls in sorted_label_scores],
                },
                signature=self.prompt_signature,
            )


class HuggingFaceClassification(ClassificationBridge[list[str], huggingface_.InferenceMode, huggingface_.Result]):
    @property
    def prompt_template(self) -> str | None:
        return "This text is about {}"

    @property
    def prompt_signature(self) -> list[str]:
        return self._labels

    @property
    def inference_mode(self) -> huggingface_.InferenceMode:
        return huggingface_.InferenceMode.default

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
    @property
    def prompt_template(self) -> str | None:
        return "This text is about {}"

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


class OutlinesClassification(ClassificationBridge[list[str], outlines_.InferenceMode, str]):
    @property
    def prompt_template(self) -> str | None:
        return f"""
        Classify the text after ======== as one or more of the following options: {",".join(self._labels)}. 
        Separate your choices with a comma.
        ========
        {{{{ text }}}}
        """

    @property
    def prompt_signature(self) -> list[str]:
        return self._labels

    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return outlines_.InferenceMode.choice

    def extract(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        return ({"text": doc.text if doc.text else None} for doc in docs)

    def integrate(self, results: Iterable[str], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            doc.results[self._task_id] = result.split(",")
        return docs

    def consolidate(self, results: Iterable[str], docs_offsets: list[tuple[int, int]]) -> Iterable[str]:
        results = list(results)

        # Determine label scores for chunks per document.
        for doc_offset in docs_offsets:
            label_counts: dict[str, int] = {label: 0 for label in self._labels}
            for rec in results[doc_offset[0] : doc_offset[1]]:
                label_counts[rec] += 1

            # Average score, sort by it in descending order.
            sorted_label_scores: list[dict[str, str | float]] = sorted(
                [{"label": label, "score": count} for label, count in label_counts.items()],
                key=lambda x: x["score"],
                reverse=True,
            )

            assert isinstance(sorted_label_scores[0]["label"], str)
            yield sorted_label_scores[0]["label"]
