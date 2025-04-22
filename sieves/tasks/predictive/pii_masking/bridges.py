"""Bridges for PII masking task."""
import abc
from collections.abc import Iterable
from functools import cached_property
from typing import Literal, TypeVar

import dspy
import jinja2
import pydantic

from sieves.data import Doc
from sieves.engines import EngineInferenceMode, dspy_, instructor_, langchain_, ollama_, outlines_, vllm_
from sieves.tasks.predictive.bridges import Bridge

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class PIIBridge(Bridge[_BridgePromptSignature, _BridgeResult, EngineInferenceMode], abc.ABC):
    """Abstract base class for PII masking bridges."""

    def __init__(
        self,
        task_id: str,
        prompt_template: str | None,
        prompt_signature_desc: str | None,
        overwrite: bool,
        mask_placeholder: str,
        pii_types: list[str] | None,
    ):
        """
        Initialize PIIBridge.

        :param task_id: Task ID.
        :param prompt_template: Custom prompt template.
        :param prompt_signature_desc: Custom prompt signature description.
        :param overwrite: Whether to overwrite text with masked text.
        :param mask_placeholder: String to replace PII with.
        :param pii_types: Types of PII to mask. If None, all common PII types will be masked.
        """
        super().__init__(
            task_id=task_id,
            prompt_template=prompt_template,
            prompt_signature_desc=prompt_signature_desc,
            overwrite=overwrite,
        )
        self._mask_placeholder = mask_placeholder
        self._pii_types = pii_types
        self._pii_entity_cls = self._create_pii_entity_cls()

    def _create_pii_entity_cls(self) -> type[pydantic.BaseModel]:
        """Creates PII entity class."""
        pii_types = self._pii_types
        PIIType = Literal[*pii_types] if pii_types else str

        class PIIEntity(pydantic.BaseModel, frozen=True):
            """PII entity."""

            entity_type: PIIType  # type: ignore[valid-type]
            text: str

        return PIIEntity


class DSPyPIIMasking(PIIBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    """DSPy bridge for PII masking."""

    @property
    def _prompt_template(self) -> str | None:
        return None

    @property
    def _prompt_signature_description(self) -> str | None:
        default_pii_types_desc = "all types of personally identifiable information"
        pii_types_desc = ", ".join(self._pii_types) if self._pii_types else default_pii_types_desc
        return (
            f"Identify and mask {pii_types_desc} in the given text. Replace each PII instance with "
            f"'{self._mask_placeholder}'."
        )

    @cached_property
    def prompt_signature(self) -> type[dspy_.PromptSignature]:
        """Define prompt signature for DSPy."""
        PIIEntity = self._pii_entity_cls

        class PIIMasking(dspy.Signature):  # type: ignore[misc]
            text: str = dspy.InputField(description="Text to mask PII from.")
            reasoning: str = dspy.OutputField(description="Reasoning about what PII was found and masked.")
            masked_text: str = dspy.OutputField(description="Text with all PII masked.")
            pii_entities: list[PIIEntity] = dspy.OutputField(description="List of PII entities that were masked.")  # type: ignore[valid-type]

        PIIMasking.__doc__ = jinja2.Template(self.prompt_signature_description).render()
        return PIIMasking

    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        """Return inference mode for DSPy engine."""
        return dspy_.InferenceMode.chain_of_thought

    def integrate(self, results: Iterable[dspy_.Result], docs: Iterable[Doc]) -> Iterable[Doc]:
        """Integrate results into docs."""
        for doc, result in zip(docs, results):
            # Store masked text and PII entities in results
            doc.results[self._task_id] = {
                "masked_text": result.masked_text,
                "pii_entities": result.pii_entities,
            }

            if self._overwrite:
                doc.text = result.masked_text

        return docs

    def consolidate(
        self, results: Iterable[dspy_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[dspy_.Result]:
        """Consolidate results from multiple chunks."""
        results = list(results)
        PIIEntity = self._pii_entity_cls

        # Merge results for each document
        for doc_offset in docs_offsets:
            doc_results = results[doc_offset[0] : doc_offset[1]]
            seen_entities: set[PIIEntity] = set()  # type: ignore[valid-type]
            entities: list[PIIEntity] = []  # type: ignore[valid-type]
            masked_texts: list[str] = []
            reasonings: list[str] = []

            for res in doc_results:
                reasonings.append(res.reasoning)
                masked_texts.append(res.masked_text)
                for entity in res.pii_entities:
                    if entity not in seen_entities:
                        entities.extend(res.pii_entities)
                        seen_entities.add(entity)

            yield dspy.Prediction.from_completions(
                {"masked_text": [" ".join(masked_texts)], "pii_entities": [entities], "reasoning": [str(reasonings)]},
                signature=self.prompt_signature,
            )


class PydanticBasedPIIMasking(PIIBridge[pydantic.BaseModel, pydantic.BaseModel, EngineInferenceMode], abc.ABC):
    """Base class for Pydantic-based PII masking bridges."""

    @property
    def _prompt_template(self) -> str | None:
        return """
        Identify and mask Personally Identifiable Information (PII) in the given text.
        {% if pii_types|length > 0 -%}
            Focus on these specific PII types: {{ pii_types|join(', ') }}.
        {% else -%}
            Mask all common types of PII such as names, addresses, phone numbers, emails, SSNs, credit 
            card numbers, etc.
        {% endif -%}
        Replace each instance of PII with "{{ mask_placeholder }}".
        
        {% if examples|length > 0 -%}
            <examples>
            {%- for example in examples %}
                <example>
                    <text>"{{ example.text }}"</text>
                    <output>
                        <reasoning>{{ example.reasoning }}</reasoning>
                        <masked_test>{{ example.masked_text }}</masked_test>
                        <pii_entities_found>{{ example.pii_entities }}</pii_entities_found>
                    </output>
                </example>
            {% endfor -%}
            </examples>
        {% endif -%}

        ========
        <text>{{ text }}</text>
        <output>
        """

    @property
    def _prompt_signature_description(self) -> str | None:
        return None

    @cached_property
    def prompt_signature(self) -> type[pydantic.BaseModel]:
        """Define prompt signature for Pydantic-based engines."""
        PIIEntity = self._pii_entity_cls

        class PIIMasking(pydantic.BaseModel, frozen=True):
            reasoning: str
            masked_text: str
            pii_entities: list[PIIEntity]  # type: ignore[valid-type]

        if self.prompt_signature_description:
            PIIMasking.__doc__ = jinja2.Template(self.prompt_signature_description).render()

        return PIIMasking

    def integrate(self, results: Iterable[pydantic.BaseModel], docs: Iterable[Doc]) -> Iterable[Doc]:
        """Integrate results into docs."""
        for doc, result in zip(docs, results):
            assert hasattr(result, "masked_text")
            assert hasattr(result, "pii_entities")
            # Store masked text and PII entities in results
            doc.results[self._task_id] = {"masked_text": result.masked_text, "pii_entities": result.pii_entities}

            if self._overwrite:
                doc.text = result.masked_text

        return docs

    def consolidate(
        self, results: Iterable[pydantic.BaseModel], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[pydantic.BaseModel]:
        """Consolidate results from multiple chunks."""
        results = list(results)
        PIIEntity = self._pii_entity_cls

        # Merge results for each document
        for doc_offset in docs_offsets:
            doc_results = results[doc_offset[0] : doc_offset[1]]
            seen_entities: set[PIIEntity] = set()  # type: ignore[valid-type]
            entities: list[PIIEntity] = []  # type: ignore[valid-type]
            masked_texts: list[str] = []
            reasonings: list[str] = []

            for res in doc_results:
                if res is None:
                    continue  # type: ignore[unreachable]

                assert hasattr(res, "reasoning")
                assert hasattr(res, "masked_text")
                assert hasattr(res, "pii_entities")

                reasonings.append(res.reasoning)
                masked_texts.append(res.masked_text)
                for entity in res.pii_entities:
                    if entity not in seen_entities:
                        entities.extend(res.pii_entities)
                        seen_entities.add(entity)

            yield self.prompt_signature(
                reasoning=str(reasonings), masked_text=" ".join(masked_texts), pii_entities=entities
            )


class OutlinesPIIMasking(PydanticBasedPIIMasking[outlines_.InferenceMode]):
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return outlines_.InferenceMode.json


class OllamaPIIMasking(PydanticBasedPIIMasking[ollama_.InferenceMode]):
    @property
    def inference_mode(self) -> ollama_.InferenceMode:
        return ollama_.InferenceMode.structured


class LangChainPIIMasking(PydanticBasedPIIMasking[langchain_.InferenceMode]):
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return langchain_.InferenceMode.structured


class InstructorPIIMasking(PydanticBasedPIIMasking[instructor_.InferenceMode]):
    @property
    def inference_mode(self) -> instructor_.InferenceMode:
        return instructor_.InferenceMode.structured


class VLLMPIIMasking(PydanticBasedPIIMasking[vllm_.InferenceMode]):
    """VLLM bridge for PII masking."""

    @property
    def inference_mode(self) -> vllm_.InferenceMode:
        """Return inference mode for VLLM engine."""
        return vllm_.InferenceMode.json
