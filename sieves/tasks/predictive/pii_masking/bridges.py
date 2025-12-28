"""Bridges for PII masking task."""

import abc
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, TypeVar, override

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
from sieves.tasks.predictive.consolidation import MultiEntityConsolidation
from sieves.tasks.predictive.schemas.pii_masking import (
    PIIEntity,
    Result,
)

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class PIIMaskingBridge(Bridge[_BridgePromptSignature, _BridgeResult, ModelWrapperInferenceMode], abc.ABC):
    """Abstract base class for PII masking bridges."""

    def __init__(
        self,
        task_id: str,
        prompt_instructions: str | None,
        mask_placeholder: str,
        pii_types: Sequence[str] | dict[str, str] | None,
        overwrite: bool,
        model_settings: ModelSettings,
        prompt_signature: type[pydantic.BaseModel],
        model_type: ModelType,
        fewshot_examples: Sequence[pydantic.BaseModel] = (),
    ):
        """Initialize PII masking bridge.

        :param task_id: Task ID.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param mask_placeholder: Placeholder for masked PII.
        :param pii_types: PII types to mask.
        :param overwrite: Whether to overwrite original text.
        :param model_settings: Settings for structured generation.
        :param prompt_signature: Unified Pydantic prompt signature.
        :param model_type: Model type.
        :param fewshot_examples: Few-shot examples.
        """
        super().__init__(
            task_id=task_id,
            prompt_instructions=prompt_instructions,
            overwrite=overwrite,
            model_settings=model_settings,
            prompt_signature=prompt_signature,
            model_type=model_type,
            fewshot_examples=fewshot_examples,
        )

        self._mask_placeholder = mask_placeholder
        self._pii_types: list[str] | None = None
        self._pii_type_descriptions: dict[str, str] = {}

        if isinstance(pii_types, dict):
            self._pii_types = list(pii_types.keys())
            self._pii_type_descriptions = pii_types
        elif pii_types is not None:
            self._pii_types = pii_types
            self._pii_type_descriptions = {}

        self._pii_entity_cls = self._create_pii_entity_cls()
        self._consolidation_strategy = MultiEntityConsolidation(extractor=self._chunk_extractor)

    @property
    @abc.abstractmethod
    def _chunk_extractor(self) -> Callable[[Any], Iterable[pydantic.BaseModel]]:
        """Return a callable that extracts a list of entities from a raw chunk result.

        :return: Extractor callable.
        """

    def _get_pii_type_descriptions(self) -> str:
        """Return a string with the PII type descriptions.

        :return: A string with the PII type descriptions.
        """
        if not self._pii_types:
            return ""

        pii_types_with_descriptions: list[str] = []
        for pii_type in self._pii_types:
            if pii_type in self._pii_type_descriptions:
                pii_types_with_descriptions.append(
                    f"  <pii_type_description>\n    <pii_type>{pii_type}</pii_type>\n    <description>"
                    f"{self._pii_type_descriptions[pii_type]}</description>\n  </pii_type_description>"
                )
            else:
                pii_types_with_descriptions.append(f"  <pii_type>{pii_type}</pii_type>")

        pii_type_desc_string = "\n".join(pii_types_with_descriptions)
        return f"<pii_type_descriptions>\n{pii_type_desc_string}\n</pii_type_descriptions>"

    def _create_pii_entity_cls(self) -> type[pydantic.BaseModel]:
        """Create PII entity class.

        :returns: PII entity class.
        """
        pii_types_list = []
        if self._pii_types:
            for pt in self._pii_types:
                pii_types_list.append(pt)
                if pt.lower() not in pii_types_list:
                    pii_types_list.append(pt.lower())
                if pt.upper() not in pii_types_list:
                    pii_types_list.append(pt.upper())

        PIIType = Literal[*pii_types_list] if pii_types_list else str  # type: ignore[invalid-type-form]

        class PIIEntityRuntime(PIIEntity, frozen=True):
            """PII entity."""

            entity_type: PIIType  # type: ignore[valid-type]

        return PIIEntityRuntime


class DSPyPIIMasking(PIIMaskingBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    """DSPy bridge for PII masking."""

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
        """Return inference mode for DSPy model wrapper."""
        return self._model_settings.inference_mode or dspy_.InferenceMode.predict

    @property
    @override
    def _chunk_extractor(self) -> Callable[[Any], Iterable[pydantic.BaseModel]]:
        return lambda res: res.pii_entities

    @override
    def integrate(self, results: Sequence[dspy_.Result], docs: list[Doc]) -> list[Doc]:
        """Integrate results into docs."""
        for doc, result in zip(docs, results):
            # Store masked text and PII entities in results
            res = Result(
                masked_text=result.masked_text,
                pii_entities=[
                    PIIEntity.model_validate(e.model_dump() if hasattr(e, "model_dump") else e)
                    for e in result.pii_entities
                ],
            )
            doc.results[self._task_id] = res

            if self._overwrite:
                doc.text = result.masked_text

        return docs

    @override
    def consolidate(
        self,
        results: Sequence[dspy_.Result],
        docs_offsets: list[tuple[int, int]],
    ) -> Sequence[dspy_.Result]:
        """Consolidate results from multiple chunks."""
        # Delegate consolidation of entities to strategy.
        consolidated_entities_all = self._consolidation_strategy.consolidate(results, docs_offsets)

        # Merge results for each document.
        consolidated_results: list[dspy_.Result] = []
        for i, (start, end) in enumerate(docs_offsets):
            doc_results = results[start:end]
            masked_texts: list[str] = []

            for res in doc_results:
                masked_texts.append(res.masked_text)

            consolidated_results.append(
                dspy.Prediction.from_completions(
                    {
                        "masked_text": [" ".join(masked_texts).strip()],
                        "pii_entities": [consolidated_entities_all[i]],
                    },
                    signature=self.prompt_signature,
                )
            )
        return consolidated_results


class PydanticPIIMasking(PIIMaskingBridge[pydantic.BaseModel, pydantic.BaseModel, ModelWrapperInferenceMode], abc.ABC):
    """Base class for Pydantic-based PII masking bridges."""

    @override
    def _validate(self) -> None:
        assert self._model_type in {ModelType.langchain, ModelType.outlines}

    @property
    @override
    def _chunk_extractor(self) -> Callable[[Any], Iterable[pydantic.BaseModel]]:
        return lambda res: res.pii_entities

    @property
    def _default_prompt_instructions(self) -> str:
        pii_type_info = self._get_pii_type_descriptions() if self._pii_type_descriptions else ""
        return (
            "Identify and mask Personally Identifiable Information (PII) in the given text.\n"
            "{%- if pii_types|length > 0 %}\n"
            "Focus on these specific PII types: {{ pii_types|join(', ') }}.\n"
            "{%- else %}\n"
            "Mask all common types of PII such as names, addresses, phone numbers, emails, SSNs, credit card numbers, "
            "etc.\n"
            "{%- endif %}\n"
            f"{pii_type_info}\n"
            f'Replace each instance of PII with "{self._mask_placeholder}".\n'
            "Provide a confidence score between 0.0 and 1.0 for each entity found."
        )

    @override
    @property
    def _prompt_conclusion(self) -> str | None:
        return "========\n\n<text>{{ text }}</text>"

    @override
    def integrate(self, results: Sequence[pydantic.BaseModel], docs: list[Doc]) -> list[Doc]:
        for doc, result in zip(docs, results):
            assert hasattr(result, "masked_text")
            assert hasattr(result, "pii_entities")
            # Store masked text and PII entities in results
            doc.results[self._task_id] = Result(
                masked_text=result.masked_text,
                pii_entities=[PIIEntity.model_validate(e) for e in result.pii_entities],
            )

            if self._overwrite:
                doc.text = result.masked_text

        return docs

    @override
    def consolidate(
        self,
        results: Sequence[pydantic.BaseModel],
        docs_offsets: list[tuple[int, int]],
    ) -> Sequence[pydantic.BaseModel]:
        assert issubclass(self.prompt_signature, pydantic.BaseModel)

        consolidated_entities_all = self._consolidation_strategy.consolidate(results, docs_offsets)
        consolidated_results: list[pydantic.BaseModel] = []

        for i, (start, end) in enumerate(docs_offsets):
            doc_results = results[start:end]
            masked_texts: list[str] = []

            for res in doc_results:
                if res is None:
                    continue  # type: ignore[unreachable]

                assert hasattr(res, "masked_text")
                masked_texts.append(res.masked_text)

            consolidated_results.append(
                self.prompt_signature(
                    masked_text=" ".join(masked_texts).strip(),
                    pii_entities=consolidated_entities_all[i],
                )
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
