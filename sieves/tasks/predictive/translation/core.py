from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeAlias

import datasets
import pydantic

from sieves.data import Doc
from sieves.engines import Engine, EngineType, dspy_, ollama_, outlines_
from sieves.serialization import Config
from sieves.tasks.predictive.core import PredictiveTask
from sieves.tasks.predictive.translation.bridges import (
    DSPyTranslation,
    InstructorTranslation,
    LangChainTranslation,
    OllamaTranslation,
    OutlinesTranslation,
)

_TaskPromptSignature: TypeAlias = pydantic.BaseModel | dspy_.PromptSignature
_TaskResult: TypeAlias = outlines_.Result | dspy_.Result | ollama_.Result
_TaskBridge: TypeAlias = (
    DSPyTranslation | InstructorTranslation | LangChainTranslation | OutlinesTranslation | OllamaTranslation
)


class FewshotExample(pydantic.BaseModel):
    text: str
    to: str
    translation: str


class Translation(PredictiveTask[_TaskPromptSignature, _TaskResult, _TaskBridge]):
    def __init__(
        self,
        to: str,
        engine: Engine,
        task_id: str | None = None,
        show_progress: bool = True,
        include_meta: bool = True,
        overwrite: bool = False,
        prompt_template: str | None = None,
        prompt_signature_desc: str | None = None,
        fewshot_examples: Iterable[FewshotExample] = (),
    ) -> None:
        """
        Initializes new PredictiveTask.
        :param to: Language to translate to.
        :param task_id: Task ID.
        :param show_progress: Whether to show progress bar for processed documents.
        :param include_meta: Whether to include meta information generated by the task.
        :param overwrite: Some tasks, e.g. anonymization or translation, output a modified version of the input text.
            If True, these tasks overwrite the original document text. If False, the result will just be stored in the
            documents' `.results` field.
        :param prompt_template: Custom prompt template. If None, task's default template is being used.
        :param prompt_signature_desc: Custom prompt signature description. If None, default will be used.
        :param fewshot_examples: Few-shot examples.
        """
        self._to = to

        super().__init__(
            engine=engine,
            task_id=task_id,
            show_progress=show_progress,
            include_meta=include_meta,
            overwrite=overwrite,
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
            EngineType.dspy: DSPyTranslation,
            EngineType.instructor: InstructorTranslation,
            EngineType.langchain: LangChainTranslation,
            EngineType.outlines: OutlinesTranslation,
            EngineType.ollama: OllamaTranslation,
        }

        try:
            bridge = bridge_types[engine_type](
                task_id=self._task_id,
                prompt_template=self._custom_prompt_template,
                prompt_signature_desc=self._custom_prompt_signature_desc,
                overwrite=self._overwrite,
                language=self._to,
            )
        except KeyError as err:
            raise KeyError(f"Engine type {engine_type} is not supported by {self.__class__.__name__}.") from err

        return bridge

    @property
    def supports(self) -> set[EngineType]:
        """
        :return set[EngineType]: Supported engine types.
        """
        return {EngineType.dspy, EngineType.instructor, EngineType.ollama, EngineType.outlines}

    @property
    def _state(self) -> dict[str, Any]:
        """
        :return dict[str, Any]: Task state.
        """
        return {
            **super()._state,
            "to": self._to,
        }

    def to_dataset(self, docs: Iterable[Doc]) -> datasets.Dataset:
        """Converts docs to Hugging Face dataset.
        :param docs: Documents to convert.
        :return datasets.Dataset: Converted dataset.
        """
        # Define metadata.
        features = datasets.Features({"text": datasets.Value("string"), "translation": datasets.Value("string")})
        info = datasets.DatasetInfo(
            description=f"Translation dataset with target language {self._to}."
            f"Generated with sieves v{Config.get_version()}.",
            features=features,
        )

        # Fetch data used for generating dataset.
        try:
            data = [(doc.text, doc.results[self._task_id]) for doc in docs]
        except KeyError as err:
            raise KeyError(f"Not all documents have results for this task with ID {self._task_id}") from err

        def generate_data() -> Iterable[dict[str, Any]]:
            """Yields results as dicts.
            :return: Results as dicts.
            """
            for text, translation in data:
                yield {"text": text, "translation": translation}

        # Create dataset.
        return datasets.Dataset.from_generator(generate_data, features=features, info=info)
