"""Bridges for question answering task."""

import abc
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar, override

import dspy
import pydantic

from sieves.data import Doc
from sieves.model_wrappers import (
    ModelType,
    ModelWrapperInferenceMode,
    dspy_,
    langchain_,
    outlines_,
)
from sieves.model_wrappers.types import ModelSettings
from sieves.tasks.predictive.bridges import Bridge
from sieves.tasks.predictive.consolidation import QAConsolidation
from sieves.tasks.predictive.schemas.question_answering import QuestionAnswer, Result

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class QuestionAnsweringBridge(Bridge[_BridgePromptSignature, _BridgeResult, ModelWrapperInferenceMode], abc.ABC):
    """Abstract base class for question answering bridges."""

    def __init__(
        self,
        task_id: str,
        prompt_instructions: str | None,
        questions: list[str],
        model_settings: ModelSettings,
        prompt_signature: type[pydantic.BaseModel],
        model_type: ModelType,
        fewshot_examples: Sequence[pydantic.BaseModel] = (),
    ):
        """Initialize question answering bridge.

        :param task_id: Task ID.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param questions: Questions to answer.
        :param model_settings: Settings for structured generation.
        :param prompt_signature: Unified Pydantic prompt signature.
        :param model_type: Model type.
        :param fewshot_examples: Few-shot examples.
        """
        super().__init__(
            task_id=task_id,
            prompt_instructions=prompt_instructions,
            overwrite=False,
            model_settings=model_settings,
            prompt_signature=prompt_signature,
            model_type=model_type,
            fewshot_examples=fewshot_examples,
        )
        self._questions = questions
        self._consolidation_strategy = QAConsolidation(questions=self._questions, extractor=self._chunk_extractor)

    @property
    @abc.abstractmethod
    def _chunk_extractor(self) -> Callable[[Any], Iterable[tuple[str, str, float | None]]]:
        """Return a callable that extracts (question, answer, score) tuples from a raw chunk result.

        :return: Extractor callable.
        """


class DSPyQuestionAnswering(QuestionAnsweringBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    """DSPy bridge for question answering."""

    @override
    def _validate(self) -> None:
        assert self._model_type == ModelType.dspy

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return ""

    @override
    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return self._model_settings.inference_mode or dspy_.InferenceMode.predict

    @property
    @override
    def _chunk_extractor(self) -> Callable[[Any], Iterable[tuple[str, str, float | None]]]:
        return lambda res: ((qa.question, qa.answer, qa.score) for qa in res.qa_pairs)

    @override
    def integrate(self, results: Sequence[dspy_.Result], docs: list[Doc]) -> list[Doc]:
        for doc, result in zip(docs, results):
            assert len(result.completions.qa_pairs) == 1
            doc.results[self._task_id] = Result(qa_pairs=result.qa_pairs)
        return docs

    @override
    def consolidate(
        self, results: Sequence[dspy_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[dspy_.Result]:
        consolidated_results_clean = self._consolidation_strategy.consolidate(results, docs_offsets)

        # Wrap back into dspy.Prediction.
        consolidated_results: list[dspy_.Result] = []
        for qa_list in consolidated_results_clean:
            consolidated_results.append(
                dspy.Prediction.from_completions(
                    {
                        "qa_pairs": [[QuestionAnswer(question=q, answer=a, score=s) for q, a, s in qa_list]],
                    },
                    signature=self.prompt_signature,
                )
            )
        return consolidated_results


class PydanticQA(QuestionAnsweringBridge[pydantic.BaseModel, pydantic.BaseModel, ModelWrapperInferenceMode], abc.ABC):
    """Base class for Pydantic-based question answering bridges."""

    @override
    def _validate(self) -> None:
        assert self._model_type in {ModelType.langchain, ModelType.outlines}

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return (
            "Use the given text to answer the following questions. Ensure you answer each question exactly once. "
            "Prefix each question with the number of the corresponding question. Also provide a confidence score "
            "between 0.0 and 1.0 for each answer."
        )

    @override
    @property
    def _prompt_conclusion(self) -> str | None:
        return "========\n<text>{{ text }}</text>"

    @property
    @override
    def _chunk_extractor(self) -> Callable[[Any], Iterable[tuple[str, str, float | None]]]:
        return lambda res: ((qa.question, qa.answer, qa.score) for qa in res.qa_pairs)

    @override
    def integrate(self, results: Sequence[pydantic.BaseModel], docs: list[Doc]) -> list[Doc]:
        for doc, result in zip(docs, results):
            assert hasattr(result, "qa_pairs")
            doc.results[self._task_id] = Result(qa_pairs=result.qa_pairs)
        return docs

    @override
    def consolidate(
        self, results: Sequence[pydantic.BaseModel], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[pydantic.BaseModel]:
        assert issubclass(self.prompt_signature, pydantic.BaseModel)

        consolidated_results_clean = self._consolidation_strategy.consolidate(results, docs_offsets)
        consolidated_results: list[pydantic.BaseModel] = []

        for qa_list in consolidated_results_clean:
            consolidated_results.append(
                self.prompt_signature(qa_pairs=[QuestionAnswer(question=q, answer=a, score=s) for q, a, s in qa_list])
            )

        return consolidated_results

    @override
    @property
    def model_type(self) -> ModelType:
        return self._model_type

    @override
    @property
    def inference_mode(self) -> outlines_.InferenceMode | langchain_.InferenceMode:
        if self._model_type == ModelType.outlines:
            return self._model_settings.inference_mode or outlines_.InferenceMode.json
        elif self._model_type == ModelType.langchain:
            return self._model_settings.inference_mode or langchain_.InferenceMode.structured

        raise ValueError(f"Unsupported model type: {self._model_type}")
