from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any, TypeAlias

import datasets
import pydantic

from sieves.data import Doc
from sieves.engines import Engine, EngineType, dspy_, ollama_, outlines_
from sieves.engines.core import EngineInferenceMode, EngineModel, EnginePromptSignature, EngineResult
from sieves.serialization import Config
from sieves.tasks.predictive.core import PredictiveTask
from sieves.tasks.predictive.information_extraction.bridges import (
    DSPyInformationExtraction,
    LangChainInformationExtraction,
    OllamaInformationExtraction,
    OutlinesInformationExtraction,
)
from sieves.tasks.utils import PydanticToHFDatasets

_TaskPromptSignature: TypeAlias = pydantic.BaseModel | dspy_.PromptSignature
_TaskInferenceMode: TypeAlias = outlines_.InferenceMode | dspy_.InferenceMode | ollama_.InferenceMode
_TaskResult: TypeAlias = outlines_.Result | dspy_.Result | ollama_.Result
_TaskBridge: TypeAlias = (
    DSPyInformationExtraction
    | LangChainInformationExtraction
    | OutlinesInformationExtraction
    | OllamaInformationExtraction
)


class TaskFewshotExample(pydantic.BaseModel):
    text: str
    reasoning: str
    entities: list[pydantic.BaseModel]


class InformationExtraction(PredictiveTask[_TaskPromptSignature, _TaskResult, _TaskBridge]):
    def __init__(
        self,
        entity_type: type[pydantic.BaseModel],
        engine: Engine[EnginePromptSignature, EngineResult, EngineModel, EngineInferenceMode],
        task_id: str | None = None,
        show_progress: bool = True,
        include_meta: bool = True,
        prompt_template: str | None = None,
        prompt_signature_desc: str | None = None,
        fewshot_examples: Iterable[TaskFewshotExample] = (),
    ) -> None:
        """
        Initializes new PredictiveTask.
        :param entity_type: Object type to extract.
        :param task_id: Task ID.
        :param show_progress: Whether to show progress bar for processed documents.
        :param include_meta: Whether to include meta information generated by the task.
        :param prompt_template: Custom prompt template. If None, task's default template is being used.
        :param prompt_signature_desc: Custom prompt signature description. If None, default will be used.
        :param fewshot_examples: Few-shot examples.
        """
        self._entity_type = entity_type
        if not self._entity_type.model_config.get("frozen", False):
            warnings.warn(
                f"Entity type provided to task {self._task_id} isn't frozen, which means that entities can't "
                f"be deduplicated. Modify entity_type to be frozen=True."
            )

        super().__init__(
            engine=engine,
            task_id=task_id,
            show_progress=show_progress,
            include_meta=include_meta,
            prompt_template=prompt_template,
            prompt_signature_desc=prompt_signature_desc,
            fewshot_examples=fewshot_examples,
        )

    def _init_bridge(self, engine_type: EngineType) -> _TaskBridge:
        """Initialize bridge.
        :param engine_type: Type of engine to initialize bridge for.
        :return _TaskBridge: Engine task bridge.
        :raises ValueError: If engine type is not supported.
        """
        bridge_types: dict[EngineType, type[_TaskBridge]] = {
            EngineType.dspy: DSPyInformationExtraction,
            EngineType.langchain: LangChainInformationExtraction,
            EngineType.outlines: OutlinesInformationExtraction,
            EngineType.ollama: OllamaInformationExtraction,
        }

        try:
            bridge = bridge_types[engine_type](
                task_id=self._task_id,
                prompt_template=self._custom_prompt_template,
                prompt_signature_desc=self._custom_prompt_signature_desc,
                entity_type=self._entity_type,
            )
        except KeyError as err:
            raise KeyError(f"Engine type {engine_type} is not supported by {self.__class__.__name__}.") from err

        return bridge

    @property
    def supports(self) -> set[EngineType]:
        """
        :return set[EngineType]: Supported engine types.
        """
        return {EngineType.outlines, EngineType.dspy, EngineType.ollama}

    @property
    def _state(self) -> dict[str, Any]:
        """
        :return dict[str, Any]: Task state.
        """
        return {
            **super()._state,
            "entity_type": self._entity_type,
        }

    def to_dataset(self, docs: Iterable[Doc]) -> datasets.Dataset:
        """
        :param docs: Documents to convert.
        :return datasets.Dataset: Converted dataset.
        """
        # Define metadata.
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "entities": datasets.Sequence(PydanticToHFDatasets.model_cls_to_features(self._entity_type)),
            }
        )
        info = datasets.DatasetInfo(
            description=f"Information extraction dataset for entity type {self._entity_type.__class__.__name__}. "
            f"Generated with sieves v{Config.get_version()}",
            features=features,
        )

        # Fetch data used for generating dataset.
        try:
            data = [
                (doc.text, [PydanticToHFDatasets.model_to_dict(res) for res in doc.results[self._task_id]])
                for doc in docs
            ]
        except KeyError as err:
            raise KeyError(f"Not all documents have results for this task with ID {self._task_id}") from err

        def generate_data() -> Iterable[dict[str, Any]]:
            """Yields results as dicts.
            :return: Results as dicts.
            """
            for text, entities in data:
                yield {"text": text, "entities": entities}

        # Create dataset.
        return datasets.Dataset.from_generator(generate_data, features=features, info=info)
