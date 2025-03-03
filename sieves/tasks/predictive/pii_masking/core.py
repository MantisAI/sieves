"""Allows masking of PII (Personally Identifiable Information) in text documents."""
from collections.abc import Iterable
from typing import Any, TypeAlias

import pydantic
from engines import dspy_

from sieves.data.doc import Doc
from sieves.engines import (
    Engine,
    EngineInferenceMode,
    EngineModel,
    EnginePromptSignature,
    EngineResult,
    EngineType,
)
from sieves.tasks.predictive.core import PredictiveTask
from sieves.tasks.predictive.pii_masking.bridges import (
    DSPyPIIMasking,
    InstructorPIIMasking,
    LangChainPIIMasking,
    OllamaPIIMasking,
    OutlinesPIIMasking,
)


class PIIEntity(pydantic.BaseModel, frozen=True):
    """PII entity."""

    entity_type: str
    text: str


_TaskPromptSignature: TypeAlias = pydantic.BaseModel | dspy_.PromptSignature
_TaskResult: TypeAlias = pydantic.BaseModel | dspy_.Result
_TaskBridge: TypeAlias = (
    DSPyPIIMasking | InstructorPIIMasking | LangChainPIIMasking | OutlinesPIIMasking | OllamaPIIMasking
)


class TaskFewshotExample(pydantic.BaseModel):
    """Example for PII masking few-shot prompting."""

    text: str
    reasoning: str
    masked_text: str
    pii_entities: list[PIIEntity]


class PIIMasking(PredictiveTask[_TaskPromptSignature, _TaskResult, _TaskBridge]):
    """Task for masking PII (Personally Identifiable Information) in text documents."""

    def __init__(
        self,
        engine: Engine[EnginePromptSignature, EngineResult, EngineModel, EngineInferenceMode],
        pii_types: list[str] | None = None,
        mask_placeholder: str = "[MASKED]",
        task_id: str | None = None,
        show_progress: bool = True,
        include_meta: bool = True,
        overwrite: bool = True,
        prompt_template: str | None = None,
        prompt_signature_desc: str | None = None,
        fewshot_examples: Iterable[TaskFewshotExample] = (),
    ) -> None:
        """
        Initialize PIIMasking task.

        :param engine: Engine to use for PII detection and masking.
        :param pii_types: Types of PII to mask. If None, all common PII types will be masked.
                         E.g., ["NAME", "EMAIL", "PHONE", "ADDRESS", "SSN", "CREDIT_CARD", "DATE_OF_BIRTH"]
        :param mask_placeholder: String to replace PII with.
        :param task_id: Task ID.
        :param show_progress: Whether to show progress bar for processed documents.
        :param include_meta: Whether to include meta information generated by the task.
        :param overwrite: Whether to overwrite original document text with masked text.
        :param prompt_template: Custom prompt template. If None, task's default template is used.
        :param prompt_signature_desc: Custom prompt signature description. If None, default will be used.
        :param fewshot_examples: Few-shot examples.
        """
        self._pii_types = pii_types
        self._mask_placeholder = mask_placeholder

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
        :return PIIBridge: Engine task bridge.
        :raises ValueError: If engine type is not supported.
        """
        bridge_types: dict[EngineType, type[_TaskBridge]] = {
            EngineType.dspy: DSPyPIIMasking,
            EngineType.instructor: InstructorPIIMasking,
            EngineType.langchain: LangChainPIIMasking,
            EngineType.outlines: OutlinesPIIMasking,
            EngineType.ollama: OllamaPIIMasking,
        }

        try:
            return bridge_types[engine_type](
                task_id=self._task_id,
                prompt_template=self._custom_prompt_template,
                prompt_signature_desc=self._custom_prompt_signature_desc,
                mask_placeholder=self._mask_placeholder,
                pii_types=self._pii_types,
                overwrite=self._overwrite,
            )
        except KeyError as err:
            raise KeyError(f"Engine type {engine_type} is not supported by {self.__class__.__name__}.") from err

    @property
    def supports(self) -> set[EngineType]:
        """
        :return set[EngineType]: Supported engine types.
        """
        return {EngineType.dspy, EngineType.instructor, EngineType.langchain, EngineType.ollama, EngineType.outlines}

    @property
    def _state(self) -> dict[str, Any]:
        """
        :return dict[str, Any]: Task state.
        """
        return {
            **super()._state,
            "pii_types": self._pii_types,
            "mask_placeholder": self._mask_placeholder,
        }

    def to_dataset(self, docs: Iterable[Doc]) -> Any:
        """Not implemented for PII masking task."""
        raise NotImplementedError("to_dataset is not implemented for PIIMasking task.")
