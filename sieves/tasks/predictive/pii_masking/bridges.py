"""Bridges for PII masking task."""

import abc
from collections.abc import Sequence
from functools import cached_property
from typing import Literal, TypeVar, override

import dspy
import jinja2
import pydantic

from sieves.data import Doc
from sieves.model_wrappers import ModelWrapperInferenceMode, dspy_, langchain_, outlines_
from sieves.model_wrappers.types import ModelSettings
from sieves.tasks.predictive.bridges import Bridge

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class PIIBridge(Bridge[_BridgePromptSignature, _BridgeResult, ModelWrapperInferenceMode], abc.ABC):
    """Abstract base class for PII masking bridges."""

    def __init__(
        self,
        task_id: str,
        prompt_instructions: str | None,
        overwrite: bool,
        mask_placeholder: str,
        pii_types: list[str] | dict[str, str] | None,
        model_settings: ModelSettings,
    ):
        """
        Initialize PIIBridge.

        :param task_id: Task ID.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param overwrite: Whether to overwrite text with masked text.
        :param mask_placeholder: String to replace PII with.
        :param pii_types: Types of PII to mask. Can be a list of PII type strings, a dict mapping PII types to
            descriptions, or None for all common PII types.
        :param model_settings: Model settings including inference_mode.
        """
        super().__init__(
            task_id=task_id,
            prompt_instructions=prompt_instructions,
            overwrite=overwrite,
            model_settings=model_settings,
        )
        self._mask_placeholder = mask_placeholder
        if isinstance(pii_types, dict):
            self._pii_types = list(pii_types.keys())
            self._pii_type_descriptions = pii_types
        elif pii_types is not None:
            self._pii_types = pii_types
            self._pii_type_descriptions = {}
        else:
            self._pii_types = None
            self._pii_type_descriptions = {}
        self._pii_entity_cls = self._create_pii_entity_cls()

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
                    f"<pii_type_description><pii_type>{pii_type}</pii_type><description>"
                    f"{self._pii_type_descriptions[pii_type]}</description></pii_type_description>"
                )
            else:
                pii_types_with_descriptions.append(pii_type)

        crlf = "\n\t\t\t"
        pii_type_desc_string = crlf + "\t" + (crlf + "\t").join(pii_types_with_descriptions)
        return f"{crlf}<pii_type_descriptions>{pii_type_desc_string}{crlf}</pii_type_descriptions>\n\t\t"

    def _create_pii_entity_cls(self) -> type[pydantic.BaseModel]:
        """Create PII entity class.

        :returns: PII entity class.
        """
        pii_types = self._pii_types
        PIIType = Literal[*pii_types] if pii_types else str  # type: ignore[invalid-type-form]

        class PIIEntity(pydantic.BaseModel, frozen=True):
            """PII entity."""

            entity_type: PIIType  # type: ignore[valid-type]
            text: str

        return PIIEntity


class DSPyPIIMasking(PIIBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    """DSPy bridge for PII masking."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        default_pii_types_desc = "all types of personally identifiable information"
        pii_types_desc = ", ".join(self._pii_types) if self._pii_types else default_pii_types_desc
        pii_type_info = self._get_pii_type_descriptions() if self._pii_type_descriptions else ""
        return (
            f"Identify and mask {pii_types_desc} in the given text. Replace each PII instance with "
            f"'{self._mask_placeholder}'.\n{pii_type_info}"
        )

    @override
    @property
    def _prompt_example_template(self) -> str | None:
        return None

    @override
    @property
    def _prompt_conclusion(self) -> str | None:
        return None

    @override
    @cached_property
    def prompt_signature(self) -> type[dspy_.PromptSignature]:
        """Define prompt signature for DSPy."""
        PIIEntity = self._pii_entity_cls

        class PIIMasking(dspy.Signature):  # type: ignore[misc]
            text: str = dspy.InputField(description="Text to mask PII from.")
            masked_text: str = dspy.OutputField(description="Text with all PII masked.")
            pii_entities: list[PIIEntity] = dspy.OutputField(description="List of PII entities that were masked.")  # type: ignore[valid-type]

        PIIMasking.__doc__ = jinja2.Template(self._prompt_instructions).render()

        return PIIMasking

    @override
    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        """Return inference mode for DSPy model wrapper."""
        return self._model_settings.inference_mode or dspy_.InferenceMode.predict

    @override
    def integrate(self, results: Sequence[dspy_.Result], docs: list[Doc]) -> list[Doc]:
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

    @override
    def consolidate(
        self, results: Sequence[dspy_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[dspy_.Result]:
        """Consolidate results from multiple chunks."""
        PIIEntity = self._pii_entity_cls

        # Merge results for each document
        consolidated_results: list[dspy_.Result] = []
        for doc_offset in docs_offsets:
            doc_results = results[doc_offset[0] : doc_offset[1]]
            seen_entities: set[PIIEntity] = set()  # type: ignore[valid-type]
            entities: list[PIIEntity] = []  # type: ignore[valid-type]
            masked_texts: list[str] = []

            for res in doc_results:
                masked_texts.append(res.masked_text)
                for entity in res.pii_entities:
                    if entity not in seen_entities:
                        entities.extend(res.pii_entities)
                        seen_entities.add(entity)

            consolidated_results.append(
                dspy.Prediction.from_completions(
                    {"masked_text": [" ".join(masked_texts)], "pii_entities": [entities]},
                    signature=self.prompt_signature,
                )
            )
        return consolidated_results


class PydanticBasedPIIMasking(PIIBridge[pydantic.BaseModel, pydantic.BaseModel, ModelWrapperInferenceMode], abc.ABC):
    """Base class for Pydantic-based PII masking bridges."""

    @property
    def _default_prompt_instructions(self) -> str:
        pii_type_info = self._get_pii_type_descriptions() if self._pii_type_descriptions else ""
        return f"""
        Identify and mask Personally Identifiable Information (PII) in the given text.
        {{%- if pii_types|length > 0 %}}
            Focus on these specific PII types: {{{{ pii_types|join(', ') }}}}.
        {{%- else %}}
            Mask all common types of PII such as names, addresses, phone numbers, emails, SSNs, credit
            card numbers, etc.
        {{%- endif %}}
        {pii_type_info}
        Replace each instance of PII with "{{{{ mask_placeholder }}}}".
        """

    @override
    @property
    def _prompt_example_template(self) -> str | None:
        return """
        {% if examples|length > 0 -%}
            <examples>
            {%- for example in examples %}
                <example>
                    <text>"{{ example.text }}"</text>
                    <output>
                        <masked_test>{{ example.masked_text }}</masked_test>
                        <pii_entities_found>{{ example.pii_entities }}</pii_entities_found>
                    </output>
                </example>
            {% endfor -%}
            </examples>
        {% endif -%}
        """

    @override
    @property
    def _prompt_conclusion(self) -> str | None:
        return """
        ========

        <text>{{ text }}</text>
        <output>
        """

    @override
    @cached_property
    def prompt_signature(self) -> type[pydantic.BaseModel]:
        """Define prompt signature for Pydantic-based model wrappers."""
        PIIEntity = self._pii_entity_cls

        class PIIMasking(pydantic.BaseModel, frozen=True):
            """PII masking output."""

            masked_text: str
            pii_entities: list[PIIEntity]  # type: ignore[valid-type]

        return PIIMasking

    @override
    def integrate(self, results: Sequence[pydantic.BaseModel], docs: list[Doc]) -> list[Doc]:
        for doc, result in zip(docs, results):
            assert hasattr(result, "masked_text")
            assert hasattr(result, "pii_entities")
            # Store masked text and PII entities in results
            doc.results[self._task_id] = {"masked_text": result.masked_text, "pii_entities": result.pii_entities}

            if self._overwrite:
                doc.text = result.masked_text

        return docs

    @override
    def consolidate(
        self, results: Sequence[pydantic.BaseModel], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[pydantic.BaseModel]:
        PIIEntity = self._pii_entity_cls

        # Merge results for each document
        consolidated_results: list[pydantic.BaseModel] = []
        for doc_offset in docs_offsets:
            doc_results = results[doc_offset[0] : doc_offset[1]]
            seen_entities: set[PIIEntity] = set()  # type: ignore[valid-type]
            entities: list[PIIEntity] = []  # type: ignore[valid-type]
            masked_texts: list[str] = []

            for res in doc_results:
                if res is None:
                    continue  # type: ignore[unreachable]

                assert hasattr(res, "masked_text")
                assert hasattr(res, "pii_entities")

                masked_texts.append(res.masked_text)
                for entity in res.pii_entities:
                    if entity not in seen_entities:
                        entities.extend(res.pii_entities)
                        seen_entities.add(entity)

            consolidated_results.append(
                self.prompt_signature(masked_text=" ".join(masked_texts), pii_entities=entities)
            )
        return consolidated_results


class OutlinesPIIMasking(PydanticBasedPIIMasking[outlines_.InferenceMode]):
    """Outlines bridge for PII masking."""

    @override
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return self._model_settings.inference_mode or outlines_.InferenceMode.json


class LangChainPIIMasking(PydanticBasedPIIMasking[langchain_.InferenceMode]):
    """LangChain bridge for PII masking."""

    @override
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return self._model_settings.inference_mode or langchain_.InferenceMode.structured
