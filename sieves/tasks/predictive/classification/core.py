from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeAlias

import datasets
import pydantic

from sieves.data import Doc
from sieves.engines import Engine, EngineType, dspy_, glix_, huggingface_, outlines_
from sieves.engines.core import EngineInferenceMode, EnginePromptSignature, EngineResult, Model
from sieves.serialization import Config
from sieves.tasks.predictive.classification.bridges import (
    DSPyClassification,
    GliXClassification,
    HuggingFaceClassification,
    LangChainClassification,
    OllamaClassification,
    OutlinesClassification,
)
from sieves.tasks.predictive.core import PredictiveTask

TaskPromptSignature: TypeAlias = list[str] | pydantic.BaseModel | dspy_.PromptSignature
TaskInferenceMode: TypeAlias = (
    outlines_.InferenceMode | dspy_.InferenceMode | huggingface_.InferenceMode | glix_.InferenceMode
)
TaskResult: TypeAlias = outlines_.Result | dspy_.Result | huggingface_.Result | glix_.Result
TaskBridge: TypeAlias = (
    DSPyClassification
    | GliXClassification
    | LangChainClassification
    | HuggingFaceClassification
    | OllamaClassification
    | OutlinesClassification
)


class TaskFewshotExample(pydantic.BaseModel):
    text: str
    reasoning: str
    confidence_per_label: dict[str, float]

    @pydantic.model_validator(mode="after")
    def check_confidence(self) -> TaskFewshotExample:
        if any([conf for conf in self.confidence_per_label.values() if not 0 <= conf <= 1]):
            raise ValueError("Confidence has to be between 0 and 1.")
        return self


class Classification(PredictiveTask[TaskPromptSignature, TaskResult, TaskInferenceMode, TaskBridge]):
    def __init__(
        self,
        labels: list[str],
        engine: Engine[EnginePromptSignature, EngineResult, Model, EngineInferenceMode],
        task_id: str | None = None,
        show_progress: bool = True,
        include_meta: bool = True,
        prompt_template: str | None = None,
        prompt_signature_desc: str | None = None,
        fewshot_examples: Iterable[TaskFewshotExample] = (),
    ) -> None:
        """
        Initializes new PredictiveTask.
        :param labels: Labels to predict.
        :param task_id: Task ID.
        :param show_progress: Whether to show progress bar for processed documents.
        :param include_meta: Whether to include meta information generated by the task.
        :param prompt_template: Custom prompt template. If None, task's default template is being used.
        :param prompt_signature_desc: Custom prompt signature description. If None, default will be used.
        :param fewshot_examples: Few-shot examples.
        """
        self._labels = labels
        super().__init__(
            engine=engine,
            task_id=task_id,
            show_progress=show_progress,
            include_meta=include_meta,
            prompt_template=prompt_template,
            prompt_signature_desc=prompt_signature_desc,
            fewshot_examples=fewshot_examples,
        )
        self._fewshot_examples: Iterable[TaskFewshotExample]

    def _init_bridge(self, engine_type: EngineType) -> TaskBridge:
        """Initialize engine task.
        :returns: Engine task.
        :raises ValueError: If engine type is not supported.
        """
        bridge_types: dict[EngineType, type[TaskBridge]] = {
            EngineType.dspy: DSPyClassification,
            EngineType.glix: GliXClassification,
            EngineType.huggingface: HuggingFaceClassification,
            EngineType.outlines: OutlinesClassification,
            EngineType.ollama: OllamaClassification,
            EngineType.langchain: LangChainClassification,
        }

        try:
            bridge = bridge_types[engine_type](
                task_id=self._task_id,
                prompt_template=self._custom_prompt_template,
                prompt_signature_desc=self._custom_prompt_signature_desc,
                labels=self._labels,
            )
        except KeyError:
            raise KeyError(f"Engine type {engine_type} is not supported by {self.__class__.__name__}.")

        return bridge

    @property
    def supports(self) -> set[EngineType]:
        return {EngineType.outlines, EngineType.dspy, EngineType.huggingface, EngineType.glix, EngineType.ollama}

    def _validate_fewshot_examples(self) -> None:
        for fs_example in self._fewshot_examples or []:
            if any([label not in self._labels for label in fs_example.confidence_per_label]) or not all(
                [label in fs_example.confidence_per_label for label in self._labels]
            ):
                raise ValueError(
                    f"Label mismatch: {self._task_id} has labels {self._labels}. Few-shot examples has "
                    f"labels {fs_example.confidence_per_label.keys()}."
                )

    @property
    def _state(self) -> dict[str, Any]:
        return {
            **super()._state,
            "labels": self._labels,
        }

    def docs_to_dataset(self, docs: Iterable[Doc]) -> datasets.Dataset:
        # Define metadata.
        features = datasets.Features(
            {"text": datasets.Value("string"), "label": datasets.Sequence(datasets.Value("float32"))}
        )
        info = datasets.DatasetInfo(
            description=f"Multi-label classification dataset with labels {self._labels}. Generated with sieves "
            f"v{Config.get_version()}",
            features=features,
        )

        # Fetch data used for generating dataset.
        labels = self._labels
        try:
            data = [(doc.text, doc.results[self._task_id]) for doc in docs]
        except KeyError as err:
            raise KeyError(f"Not all documents have results for this task with ID {self._task_id}") from err

        def generate_data() -> Iterable[dict[str, Any]]:
            """Yields results as dicts.
            :return: Results as dicts.
            """
            for text, result in data:
                scores = {label_score[0]: label_score[1] for label_score in result}
                yield {"text": text, "label": [scores[label] for label in labels]}

        # Create dataset.
        return datasets.Dataset.from_generator(generate_data, features=features, info=info)
