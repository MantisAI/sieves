from collections.abc import Iterable
from typing import Any, Literal, TypeAlias

import dspy

from sieves.data import Doc
from sieves.engines import dspy_, glix_, huggingface_, outlines_

TaskPromptSignature: TypeAlias = list[str] | type[dspy_.PromptSignature]  # type: ignore[valid-type]
TaskInferenceMode: TypeAlias = (
    outlines_.InferenceMode | dspy_.InferenceMode | huggingface_.InferenceMode | glix_.InferenceMode
)
TaskResult: TypeAlias = outlines_.Result | dspy_.Result | huggingface_.Result | glix_.Result


class _EngineTask:
    def __init__(self, task_id: str, labels: list[str]):
        self._task_id = task_id
        self._labels = labels


class DSPyClassification(_EngineTask):
    @property
    def prompt_template(self) -> str | None:
        return None

    def create_prompt_signature(self) -> type[dspy.Signature]:
        labels = self._labels
        # Dynamically create Literal as output type.
        LabelType = Literal[*labels]  # type: ignore[valid-type]

        class TextClassification(dspy.Signature):  # type: ignore[misc]
            text: str = dspy.InputField()
            labels: LabelType = dspy.OutputField()

        return TextClassification

    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return dspy_.InferenceMode.predict

    def extract_from_docs(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        return ({"text": doc.text[:256] if doc.text else None} for doc in docs)

    def integrate_into_docs(self, results: Iterable[dspy_.Result], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            doc.results[self._task_id] = result.completions.labels
        return docs


class HuggingFaceClassification(_EngineTask):
    @property
    def prompt_template(self) -> str | None:
        return "This text is about {}"

    def create_prompt_signature(self) -> list[str]:
        return self._labels

    @property
    def inference_mode(self) -> huggingface_.InferenceMode:
        return huggingface_.InferenceMode.default

    def extract_from_docs(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        return ({"text": doc.text[:256] if doc.text else None} for doc in docs)

    def integrate_into_docs(self, results: Iterable[TaskResult], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            doc.results[self._task_id] = [(label, score) for label, score in zip(result["labels"], result["scores"])]  # type: ignore[index,arg-type]
        return docs


class GliXClassification(_EngineTask):
    @property
    def prompt_template(self) -> str | None:
        return "This text is about {}"

    def create_prompt_signature(self) -> list[str]:
        return self._labels

    @property
    def inference_mode(self) -> glix_.InferenceMode:
        return glix_.InferenceMode.gliclass

    def extract_from_docs(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        return ({"text": doc.text[:256] if doc.text else None} for doc in docs)

    def integrate_into_docs(self, results: Iterable[TaskResult], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            doc.results[self._task_id] = [
                (res["label"], res["score"]) for res in sorted(result, key=lambda x: x["score"], reverse=True)
            ]
        return docs


class OutlinesClassification(_EngineTask):
    @property
    def prompt_template(self) -> str | None:
        return f"""
        Classify the text after ======== as one or more of the following options: {",".join(self._labels)}. 
        Separate your choices with a comma.
        ========
        {{{{ text }}}}
        """

    def create_prompt_signature(self) -> list[str]:
        return self._labels

    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return outlines_.InferenceMode.choice

    def extract_from_docs(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        return ({"text": doc.text[:256] if doc.text else None} for doc in docs)

    def integrate_into_docs(self, results: Iterable[TaskResult], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            doc.results[self._task_id] = result.split(",")  # type: ignore[union-attr]
        return docs
