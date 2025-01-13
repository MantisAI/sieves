from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias

import pydantic
from tasks.predictive.classification.bridges import (
    ClassificationBridge,
    DSPyClassification,
    GliXClassification,
    HuggingFaceClassification,
    OutlinesClassification,
)

from sieves.engines import Engine, EngineType, dspy_, glix_, huggingface_, outlines_
from sieves.engines.core import EngineInferenceMode, EnginePromptSignature, EngineResult, Model
from sieves.tasks.core import Bridge, PredictiveTask

TaskPromptSignature: TypeAlias = list[str] | type[pydantic.BaseModel] | type[dspy_.PromptSignature]  # type: ignore[valid-type]
TaskInferenceMode: TypeAlias = (
    outlines_.InferenceMode | dspy_.InferenceMode | huggingface_.InferenceMode | glix_.InferenceMode
)
TaskResult: TypeAlias = outlines_.Result | dspy_.Result | huggingface_.Result | glix_.Result


class TaskFewshotExample(pydantic.BaseModel):
    text: str
    confidence_per_label: dict[str, float]

    @pydantic.model_validator(mode="after")
    def check_confidence(self) -> TaskFewshotExample:
        if any([conf for conf in self.confidence_per_label.values() if not 0 <= conf <= 1]):
            raise ValueError("Confidence has to be between 0 and 1.")
        return self


class Classification(PredictiveTask[TaskPromptSignature, TaskResult, Model, TaskInferenceMode, TaskFewshotExample]):
    def __init__(
        self,
        labels: list[str],
        engine: Engine[EnginePromptSignature, EngineResult, Model, EngineInferenceMode],
        task_id: str | None = None,
        show_progress: bool = True,
        include_meta: bool = True,
        prompt_template: str | None = None,
        fewshot_examples: Iterable[TaskFewshotExample] = (),
    ) -> None:
        """
        Initializes new PredictiveTask.
        :param labels: Labels to predict.
        :param task_id: Task ID.
        :param show_progress: Whether to show progress bar for processed documents.
        :param include_meta: Whether to include meta information generated by the task.
        :param prompt_template: Custom prompt template. If None, task's default template is being used.
        """
        self._labels = labels
        super().__init__(
            engine=engine,
            task_id=task_id,
            show_progress=show_progress,
            include_meta=include_meta,
            prompt_template=prompt_template,
            fewshot_examples=fewshot_examples,
        )

    def _init_bridge(self, engine_type: EngineType) -> ClassificationBridge:
        """Initialize engine task.
        :returns: Engine task.
        :raises ValueError: If engine type is not supported.
        """
        bridge_types = {
            EngineType.dspy: DSPyClassification,
            EngineType.glix: GliXClassification,
            EngineType.huggingface: HuggingFaceClassification,
            EngineType.outlines: OutlinesClassification,
        }

        try:
            bridge = bridge_types[engine_type](
                self._task_id, self._labels, custom_prompt_template=self._custom_prompt_template
            )
        except KeyError:
            raise KeyError(f"Engine type {engine_type} is not supported by Classification task.")

        assert isinstance(bridge, Bridge)
        return bridge

    @property
    def supports(self) -> set[EngineType]:
        return {EngineType.outlines, EngineType.dspy, EngineType.huggingface, EngineType.glix}

    def _validate_fewshot_examples(self) -> None:
        for fs_example in self._fewshot_examples or []:
            if any([label not in self._labels for label in fs_example.confidence_per_label]) or not all(
                [label in fs_example.confidence_per_label for label in self._labels]
            ):
                raise ValueError(
                    f"Label mismatch: {self._task_id} has labels {self._labels}. Few-shot examples has "
                    f"labels {fs_example.confidence_per_label.keys()}."
                )
