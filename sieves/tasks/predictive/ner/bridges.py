"""Bridges for NER task."""

import abc
import re
from collections.abc import Sequence
from functools import cached_property
from typing import Any, Literal, TypeVar, override

import dspy
import jinja2
import pydantic

from sieves.data import Doc
from sieves.model_wrappers import ModelWrapperInferenceMode, dspy_, langchain_, outlines_
from sieves.model_wrappers.types import ModelSettings
from sieves.tasks.predictive.bridges import Bridge, Entities, Entity

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class NERBridge(Bridge[_BridgePromptSignature, _BridgeResult, ModelWrapperInferenceMode], abc.ABC):
    """Abstract base class for NER bridges."""

    def __init__(
        self,
        entities: list[str] | dict[str, str],
        task_id: str,
        prompt_instructions: str | None,
        model_settings: ModelSettings,
    ):
        """Initialize NERBridge.

        :param entities: Entity types to extract. Can be a list of entity type strings, or a dict mapping entity types
            to descriptions.
        :param task_id: Task ID.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param model_settings: Model settings including inference_mode.
        """
        super().__init__(
            task_id=task_id,
            prompt_instructions=prompt_instructions,
            overwrite=False,
            model_settings=model_settings,
        )
        if isinstance(entities, dict):
            self._entities = list(entities.keys())
            self._entity_descriptions = entities
        else:
            self._entities = entities
            self._entity_descriptions = {}

    def _get_entity_descriptions(self) -> str:
        """Return a string with the entity descriptions.

        :return: A string with the entity descriptions.
        """
        entities_with_descriptions: list[str] = []
        for entity in self._entities:
            if entity in self._entity_descriptions:
                entities_with_descriptions.append(
                    f"<entity_description><entity>{entity}</entity><description>"
                    f"{self._entity_descriptions[entity]}</description></entity_description>"
                )
            else:
                entities_with_descriptions.append(entity)

        crlf = "\n\t\t\t"
        entity_desc_string = crlf + "\t" + (crlf + "\t").join(entities_with_descriptions)
        return f"{crlf}<entity_descriptions>{entity_desc_string}{crlf}</entity_descriptions>\n\t\t"

    @override
    def extract(self, docs: Sequence[Doc]) -> Sequence[dict[str, Any]]:
        """Extract all values from doc instances that are to be injected into the prompts.

        Overriding the default implementation to include the entity types in the extracted values.
        :param docs: Docs to extract values from.
        :return: All values from doc instances that are to be injected into the prompts as a sequence.
        """
        return [{"text": doc.text if doc.text else None, "entity_types": self._entities} for doc in docs]

    @staticmethod
    def _find_entity_positions(
        doc_text: str,
        result: _BridgeResult,
    ) -> list[Entity]:
        """Find all positions of an entity in a document.

        :param doc_text: The text of the document.
        :param result: The result of the model.
        :return: The list of entities with start/end indices.
        """
        doc_text_lower = doc_text.lower()
        # Create a new result with the same structure as the original
        new_entities: list[Entity] = []

        # Track entities by position to avoid duplicates
        entities_by_position: dict[tuple[int, int], Entity] = {}
        context_list: list[str] = []

        entities_list = getattr(result, "entities", [])
        for entity_with_context in entities_list:
            # Skip if there is no entity
            if not entity_with_context:
                continue

            # Get the entity and context texts from the model
            entity_text = getattr(entity_with_context, "text", "")
            context = getattr(entity_with_context, "context", None)
            entity_type = getattr(entity_with_context, "entity_type", "")

            if not entity_text:
                continue

            if context is None:
                new_entities.append(
                    Entity(
                        text=entity_text,
                        start=-1,
                        end=-1,
                        entity_type=entity_type,
                    )
                )
                continue

            entity_text_lower = entity_text.lower()
            context_lower = context.lower() if context else ""
            # Create a list of the unique contexts
            # Avoid adding duplicates as entities witht he same context would be captured twice
            if context_lower not in context_list:
                context_list.append(context_lower)
            else:
                continue
            # Find all occurrences of the context in the document using regex
            context_positions = re.finditer(re.escape(context_lower), doc_text_lower)

            # For each context position that was found (usually is just one), find the entity within that context
            for match in context_positions:
                context_start = match.start()
                entity_start_in_context = context_lower.find(entity_text_lower)

                if entity_start_in_context >= 0:
                    start = context_start + entity_start_in_context
                    end = start + len(entity_text)

                    # Create a new entity with start/end indices
                    new_entity = Entity(
                        text=doc_text[start:end],
                        start=start,
                        end=end,
                        entity_type=entity_type,
                    )

                    # Only add if this exact position hasn't been filled yet
                    position_key = (start, end)
                    if position_key not in entities_by_position:
                        entities_by_position[position_key] = new_entity
                        new_entities.append(new_entity)

        return sorted(new_entities, key=lambda x: x.start)

    @override
    def integrate(self, results: Sequence[_BridgeResult], docs: list[Doc]) -> list[Doc]:
        docs_list = list(docs)
        results_list = list(results)

        for doc, result in zip(docs_list, results_list):
            # Get the original text from the document
            doc_text = doc.text or ""
            if hasattr(result, "entities"):
                # Process entities from result if available
                entities_with_position = self._find_entity_positions(doc_text, result)
                # Create a new result with the updated entities
                new_result = Entities(text=doc_text, entities=entities_with_position)
                doc.results[self._task_id] = new_result
            else:
                # Default empty result
                doc.results[self._task_id] = Entities(text=doc_text, entities=[])

        return docs_list


class DSPyNER(NERBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    """DSPy bridge for NER."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        entity_info = self._get_entity_descriptions() if self._entity_descriptions else ""
        return f"""
        A named entity recognition result that represents named entities from the provided text.
        For each entity found it includes:
        - exact text of the entity
        - a context string that contains the exact entity text along with a few surrounding words
          (two or three surronding words). The context includes the entity text itself.
        - if the same entity appears multiple times in the text, each occurrence is listed separately with its
        own context
        - the entity type from the provided list of entity types. Only entities of the specified types are included.
        {entity_info}
        """

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
        entity_types = self._entities
        LiteralType = Literal[*entity_types]  # type: ignore[valid-type]

        class Entity(dspy.Signature):  # type: ignore[misc]
            text: str = dspy.OutputField(
                description="The extracted entity text, if the same entity appears multiple times in the text, "
                "includes each occurrence separately."
            )
            context: str = dspy.OutputField(
                description="A context string that MUST include the exact entity text. The context should include "
                "the entity and a few surrounding words (two or three surrounding words). IMPORTANT: The entity text "
                "MUST be present in the context string exactly as it appears in the text."
            )
            entity_type: LiteralType = dspy.OutputField(description="The type of entity")

        class Prediction(dspy.Signature):  # type: ignore[misc]
            text: str = dspy.InputField(description="Text to extract entities from")
            entity_types: list[str] = dspy.InputField(description="List of entity types to extract")

            entities: list[Entity] = dspy.OutputField(
                description="List of entities found in the text. If the same entity appears multiple times "
                "in different contexts, include each occurrence separately."
            )

        Prediction.__doc__ = jinja2.Template(self._prompt_instructions).render()

        return Prediction

    @override
    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return self._model_settings.inference_mode or dspy_.InferenceMode.predict

    @override
    def consolidate(
        self, results: Sequence[dspy_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[dspy_.Result]:
        # Process each document (which may consist of multiple chunks)
        consolidated_results: list[dspy_.Result] = []
        for doc_offset in docs_offsets:
            doc_results = results[doc_offset[0] : doc_offset[1]]

            # Combine all entities from all chunks
            all_entities: list[Entity] = []

            # Process each chunk for this document
            for chunk_result in doc_results:
                if chunk_result is None:
                    continue

                if not hasattr(chunk_result, "entities") or not chunk_result.entities:
                    continue

                # Process entities in this chunk
                for entity in chunk_result.entities:
                    all_entities.append(entity)

            # Create a consolidated result for this document
            consolidated_results.append(
                dspy.Prediction.from_completions({"entities": [all_entities]}, signature=self.prompt_signature)
            )
        return consolidated_results


class PydanticBasedNER(NERBridge[pydantic.BaseModel, pydantic.BaseModel, ModelWrapperInferenceMode], abc.ABC):
    """Base class for Pydantic-based NER bridges."""

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        entity_info = self._get_entity_descriptions() if self._entity_descriptions else ""
        return f"""
        Your goal is to extract named entities from the text. Only extract entities of the specified types:
        {{{{ entity_types }}}}.
        {entity_info}

        For each entity:
        - Extract the exact text of the entity
        - Include a SHORT context string that contains ONLY the entity and AT MOST 3 words before and 3 words after it.
          DO NOT include the entire text as context. DO NOT include words that are not present in the original text
          as introductory words (Eg. 'Text:' before context string).
        - Specify which type of entity it is (must be one of the provided entity types)

        IMPORTANT:
        - If the same entity appears multiple times in the text, extract each occurrence separately with its own context
        """

    @override
    @property
    def _prompt_example_template(self) -> str | None:
        return """
        {% if examples|length > 0 -%}
            <examples>
            {%- for example in examples %}
                <example>
                    <text>{{ example.text }}</text>
                    <entity_types>{{ entity_types }}</entity_types>
                    <entities>
                        {%- for entity in example.entities %}
                        <entity>
                            <text>{{ entity.text }}</text>
                            <context>{{ entity.context }}</context>
                            <entity_type>{{ entity.entity_type }}</entity_type>
                        </entity>
                        {%- endfor %}
                    </entities>
                </example>
            {% endfor -%}
            </examples>
        {% endif %}
        """

    @override
    @property
    def _prompt_conclusion(self) -> str | None:
        return """
        ===========

        <text>{{ text }}</text>
        <entity_types>{{ entity_types }}</entity_types>
        <entities>
        """

    @override
    @cached_property
    def prompt_signature(self) -> type[pydantic.BaseModel]:
        entity_types = self._entities
        LiteralType = Literal[*entity_types]  # type: ignore[valid-type]

        class _EntityWithContext(pydantic.BaseModel):
            text: str
            context: str
            entity_type: LiteralType

        class Prediction(pydantic.BaseModel):
            """NER prediction."""

            entities: list[_EntityWithContext] = []

        return Prediction

    @override
    def consolidate(
        self, results: Sequence[pydantic.BaseModel], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[pydantic.BaseModel]:
        # Process each document (which may consist of multiple chunks)
        consolidated_results: list[pydantic.BaseModel] = []
        for doc_offset in docs_offsets:
            doc_results = results[doc_offset[0] : doc_offset[1]]

            # Combine all entities from all chunks
            all_entities: list[dict[str, Any]] = []

            # Process each chunk for this document
            for chunk_result in doc_results:
                if chunk_result is None:
                    continue

                if not hasattr(chunk_result, "entities") or not chunk_result.entities:
                    continue

                # Process entities in this chunk
                for entity in chunk_result.entities:
                    # We just need to combine all entities from all chunks
                    all_entities.append(entity)

            # Create a consolidated result for this document - instantiate the class with entities
            consolidated_results.append(self.prompt_signature(entities=all_entities))
        return consolidated_results


class OutlinesNER(PydanticBasedNER[outlines_.InferenceMode]):
    """Outlines bridge for NER."""

    @override
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return self._model_settings.inference_mode or outlines_.InferenceMode.json


class LangChainNER(PydanticBasedNER[langchain_.InferenceMode]):
    """LangChain bridge for NER."""

    @override
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return self._model_settings.inference_mode or langchain_.InferenceMode.structured
