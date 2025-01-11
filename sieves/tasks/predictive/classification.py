from collections.abc import Iterable
from typing import Any, Literal, TypeAlias

import dspy

from sieves.data import Doc
from sieves.engines import DSPy, Engine, EngineType, GliX, HuggingFace, Outlines, dspy_, glix_, huggingface_, outlines_
from sieves.engines.core import InferenceMode, Model, PromptSignature, Result
from sieves.tasks.core import PredictiveTask

TaskPromptSignature: TypeAlias = list[str] | dspy_.PromptSignature
TaskInferenceMode: TypeAlias = (
    outlines_.InferenceMode | dspy_.InferenceMode | huggingface_.InferenceMode | glix_.InferenceMode
)
TaskResult: TypeAlias = outlines_.Result | dspy_.Result | huggingface_.Result | glix_.Result


class Classification(PredictiveTask[TaskPromptSignature, TaskResult, Model, TaskInferenceMode]):
    def __init__(
        self,
        labels: list[str],
        engine: Engine[PromptSignature, Result, Model, InferenceMode],
        task_id: str | None = None,
        show_progress: bool = True,
        include_meta: bool = True,
    ) -> None:
        """
        Initializes new PredictiveTask.
        :param labels: Labels to predict.
        :param task_id: Task ID.
        :param show_progress: Whether to show progress bar for processed documents.
        :param include_meta: Whether to include meta information generated by the task.
        """
        self._labels = labels
        self._engine = engine
        self._prompt_signature = self._create_prompt_signature()
        super().__init__(engine=engine, task_id=task_id, show_progress=show_progress, include_meta=include_meta)

    @property
    def supports(self) -> set[EngineType]:
        return {EngineType.outlines, EngineType.dspy, EngineType.huggingface}

    @property
    def _inference_mode(self) -> TaskInferenceMode:
        match self._engine:
            case Outlines():
                return outlines_.InferenceMode.choice
            case DSPy():
                return dspy_.InferenceMode.predict
            case HuggingFace():
                return huggingface_.InferenceMode.default
            case GliX():
                return glix_.InferenceMode.gliclass
            case _:
                raise ValueError(f"Unsupported engine type: {type(self._engine)}")

    @property
    def prompt_template(self) -> str | None:
        match self._engine:
            case Outlines():
                return f"""
                Classify the text after ======== as one or more of the following options: {",".join(self._labels)}. 
                Separate your choices with a comma.
                ========
                {{{{ text }}}}
                """
            case DSPy():
                return None
            case HuggingFace():
                return "This text is about {}"
            case GliX():
                return None
            case _:
                raise ValueError(f"Unsupported engine type: {type(self._engine)}")

    def _create_prompt_signature(self) -> TaskPromptSignature:
        match self._engine:
            case Outlines():
                return self._labels
            case DSPy():
                labels = self._labels
                # Dynamically create Literal as output type.
                LabelType = Literal[*labels]  # type: ignore[valid-type]

                class TextClassification(dspy.Signature):  # type: ignore[misc]
                    text: str = dspy.InputField()
                    labels: LabelType = dspy.OutputField()

                return TextClassification
            case HuggingFace():
                return self._labels
            case GliX():
                return self._labels
            case _:
                raise ValueError(f"Unsupported engine type: {type(self._engine)}")

    def _extract_from_docs(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        # todo Remove slicing once we have chunking support.
        match self._engine:
            case Outlines():
                return ({"text": doc.text[:256] if doc.text else None} for doc in docs)
            case DSPy():
                return ({"text": doc.text[:256] if doc.text else None} for doc in docs)
            case HuggingFace():
                return ({"text": doc.text[:256] if doc.text else None} for doc in docs)
            case GliX():
                return ({"text": doc.text[:256] if doc.text else None} for doc in docs)
            case _:
                raise ValueError(f"Unsupported engine type: {type(self._engine)}")

    def _integrate_into_docs(self, results: Iterable[TaskResult], docs: Iterable[Doc]) -> Iterable[Doc]:
        match self._engine:
            # mypy ignore[union-attr] directives are due to mypy not understanding properties of Union types (such as
            # TaskResult) properly.
            case Outlines():
                for doc, result in zip(docs, results):
                    doc.results[self.id] = result.split(",")  # type: ignore[union-attr]
            case DSPy():
                for doc, result in zip(docs, results):
                    doc.results[self.id] = result.completions.labels  # type: ignore[union-attr]
            case HuggingFace():
                for doc, result in zip(docs, results):
                    doc.results[self.id] = [(label, score) for label, score in zip(result["labels"], result["scores"])]  # type: ignore[index,arg-type]
            case GliX():
                for doc, result in zip(docs, results):
                    doc.results[self.id] = [
                        (res["label"], res["score"]) for res in sorted(result, key=lambda x: x["score"], reverse=True)
                    ]
            case _:
                raise ValueError(f"Unsupported engine type: {type(self._engine)}")

        return docs
