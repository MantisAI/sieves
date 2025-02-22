from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeAlias

import datasets
import pydantic

from sieves.data import Doc
from sieves.engines import Engine, EngineType, dspy_
from sieves.engines.core import EngineInferenceMode, EngineModel, EnginePromptSignature, EngineResult
from sieves.serialization import Config
from sieves.tasks.predictive.core import PredictiveTask
from sieves.tasks.predictive.sentiment_analysis.bridges import (
    DSPySentimentAnalysis,
    InstructorSentimentAnalysis,
    LangChainSentimentAnalysis,
    OllamaSentimentAnalysis,
    OutlinesSentimentAnalysis,
)

_TaskPromptSignature: TypeAlias = pydantic.BaseModel | dspy_.PromptSignature
_TaskResult: TypeAlias = str | pydantic.BaseModel | dspy_.Result
_TaskBridge: TypeAlias = (
    DSPySentimentAnalysis
    | InstructorSentimentAnalysis
    | LangChainSentimentAnalysis
    | OllamaSentimentAnalysis
    | OutlinesSentimentAnalysis
)


class TaskFewshotExample(pydantic.BaseModel):
    text: str
    reasoning: str
    sentiment_per_aspect: dict[str, float]

    @pydantic.model_validator(mode="after")
    def check_confidence(self) -> TaskFewshotExample:
        assert "overall" in self.sentiment_per_aspect, ValueError(
            "'overall' score has to be given in `sentiment_per_aspect` dict."
        )
        if any([conf for conf in self.sentiment_per_aspect.values() if not 0 <= conf <= 1]):
            raise ValueError("Sentiment score has to be between 0 and 1.")
        return self


class SentimentAnalysis(PredictiveTask[_TaskPromptSignature, _TaskResult, _TaskBridge]):
    def __init__(
        self,
        engine: Engine[EnginePromptSignature, EngineResult, EngineModel, EngineInferenceMode],
        aspects: tuple[str, ...] = tuple(),
        task_id: str | None = None,
        show_progress: bool = True,
        include_meta: bool = True,
        prompt_template: str | None = None,
        prompt_signature_desc: str | None = None,
        fewshot_examples: Iterable[TaskFewshotExample] = (),
    ) -> None:
        """
        Initializes new SentimentAnalysis task.
        :param aspects: Aspects to consider in sentiment analysis. Overall sentiment will always be determined. If
            empty, only overall sentiment will be determined.
        :param task_id: Task ID.
        :param show_progress: Whether to show progress bar for processed documents.
        :param include_meta: Whether to include meta information generated by the task.
        :param prompt_template: Custom prompt template. If None, task's default template is being used.
        :param prompt_signature_desc: Custom prompt signature description. If None, default will be used.
        :param fewshot_examples: Few-shot examples.
        """
        self._aspects = tuple(sorted(set(aspects) | {"overall"}))
        super().__init__(
            engine=engine,
            task_id=task_id,
            show_progress=show_progress,
            include_meta=include_meta,
            overwrite=False,
            prompt_template=prompt_template,
            prompt_signature_desc=prompt_signature_desc,
            fewshot_examples=fewshot_examples,
        )
        self._fewshot_examples: Iterable[TaskFewshotExample]

    def _init_bridge(self, engine_type: EngineType) -> _TaskBridge:
        """Initialize bridge.
        :return: Engine task.
        :raises ValueError: If engine type is not supported.
        """
        bridge_types: dict[EngineType, type[_TaskBridge]] = {
            EngineType.dspy: DSPySentimentAnalysis,
            EngineType.instructor: InstructorSentimentAnalysis,
            EngineType.outlines: OutlinesSentimentAnalysis,
            EngineType.ollama: OllamaSentimentAnalysis,
            EngineType.langchain: LangChainSentimentAnalysis,
        }

        try:
            bridge_type = bridge_types[engine_type]

            return bridge_type(
                task_id=self._task_id,
                prompt_template=self._custom_prompt_template,
                prompt_signature_desc=self._custom_prompt_signature_desc,
                aspects=self._aspects,
            )
        except KeyError as err:
            raise KeyError(f"Engine type {engine_type} is not supported by {self.__class__.__name__}.") from err

    @property
    def supports(self) -> set[EngineType]:
        return {
            EngineType.dspy,
            EngineType.instructor,
            EngineType.langchain,
            EngineType.ollama,
            EngineType.outlines,
        }

    def _validate_fewshot_examples(self) -> None:
        for fs_example in self._fewshot_examples or []:
            if any([aspect not in self._aspects for aspect in fs_example.sentiment_per_aspect]) or not all(
                [label in fs_example.sentiment_per_aspect for label in self._aspects]
            ):
                raise ValueError(
                    f"Aspect mismatch: {self._task_id} has aspects {self._aspects}. Few-shot examples have "
                    f"aspects {fs_example.sentiment_per_aspect.keys()}."
                )

    @property
    def _state(self) -> dict[str, Any]:
        return {
            **super()._state,
            "aspects": self._aspects,
        }

    def to_dataset(self, docs: Iterable[Doc]) -> datasets.Dataset:
        # Define metadata.
        features = datasets.Features(
            {"text": datasets.Value("string"), "aspect": datasets.Sequence(datasets.Value("float32"))}
        )
        info = datasets.DatasetInfo(
            description=f"Aspect-based sentiment analysis dataset with aspects {self._aspects}. Generated with sieves "
            f"v{Config.get_version()}.",
            features=features,
        )

        # Fetch data used for generating dataset.
        aspects = self._aspects
        try:
            data = [(doc.text, doc.results[self._task_id]) for doc in docs]
        except KeyError as err:
            raise KeyError(f"Not all documents have results for this task with ID {self._task_id}") from err

        def generate_data() -> Iterable[dict[str, Any]]:
            """Yields results as dicts.
            :return: Results as dicts.
            """
            for text, result in data:
                scores = {sent_score[0]: sent_score[1] for sent_score in result}
                yield {"text": text, "aspect": [scores[aspect] for aspect in aspects]}

        # Create dataset.
        return datasets.Dataset.from_generator(generate_data, features=features, info=info)
