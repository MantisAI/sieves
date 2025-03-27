import abc
import re
from collections.abc import Iterable
from functools import cached_property
from typing import Any, Literal, TypeVar

import dspy
import jinja2
import pydantic

from sieves.data import Doc
from sieves.engines import EngineInferenceMode, dspy_, glix_, instructor_, langchain_, ollama_, outlines_
from sieves.tasks.predictive.bridges import Bridge

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class Entity(pydantic.BaseModel):
    text: str
    start: int | None
    end: int | None
    entity_type: str


class Entities(pydantic.BaseModel):
    entities: list[Entity]
    text: str


class NERBridge(Bridge[_BridgePromptSignature, _BridgeResult, EngineInferenceMode], abc.ABC):
    def __init__(
        self,
        entities: list[str],
        task_id: str,
        prompt_template: str | None,
        prompt_signature_desc: str | None,
    ):
        """
        Initializes NERBridge.
        :param task_id: Task ID.
        :param prompt_template: Custom prompt template.
        :param prompt_signature_desc: Custom prompt signature description.
        """
        super().__init__(
            task_id=task_id,
            prompt_template=prompt_template,
            prompt_signature_desc=prompt_signature_desc,
            overwrite=False,
        )
        self._entities = entities

    def extract(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        return ({"text": doc.text if doc.text else None, "entity_types": self._entities} for doc in docs)

    def _find_entity_positions(
        self,
        doc_text: str,
        result: _BridgeResult,
        new_entities: list[Entity],
    ) -> list[Entity]:
        """
        Find all positions of an entity in a document.
        :param doc_text: The text of the document.
        :param result: The result of the model.
        :param new_entities: The list of entities to be updated.
        :return: The list of entities with start/end indices.
        """
        doc_text_lower = doc_text.lower()

        entities_list = getattr(result, "entities", [])
        for entity_with_context in entities_list:
            # Skip if there is no entity
            if not entity_with_context:
                continue

            # Get the entity and context texts from the model
            entity_text = getattr(entity_with_context, "text", "")
            context = getattr(entity_with_context, "context", "")
            entity_type = getattr(entity_with_context, "entity_type", "")

            if not entity_text:
                continue

            entity_text_lower = entity_text.lower()
            context_lower = context.lower() if context else ""

            # First try to find the entity in the context
            if context and entity_text in context:
                # Find all occurrences of the context in the document using regex
                matches = re.findall(re.escape(context_lower), doc_text_lower)
                context_positions = [doc_text_lower.find(match) for match in matches]

                # For each context position that was found (usually is just one), find the entity within that context
                for context_start in context_positions:
                    entity_start_in_context = context_lower.find(entity_text_lower)

                    if entity_start_in_context >= 0:
                        start = context_start + entity_start_in_context
                        end = start + len(entity_text)

                        # Create a new entity with start/end indices
                        new_entity = Entity(text=doc_text[start:end], start=start, end=end, entity_type=entity_type)
                        new_entities.append(new_entity)
            else:
                # If context approach fails, find all occurrences of the entity directly
                entity_positions: list[int] = []
                pos = 0
                while True:
                    pos = doc_text_lower.find(entity_text_lower, pos)
                    if pos == -1:
                        break
                    entity_positions.append(pos)
                    pos += 1  # Move past the current position to find the next occurrence

                # Create entities for each occurrence
                for start in entity_positions:
                    end = start + len(entity_text)
                    new_entity = Entity(
                        text=doc_text[start:end],  # Use the exact text from the document
                        start=start,
                        end=end,
                        entity_type=entity_type,
                    )
                    new_entities.append(new_entity)

        return new_entities

    def integrate(self, results: Iterable[_BridgeResult], docs: Iterable[Doc]) -> Iterable[Doc]:
        docs_list = list(docs)
        results_list = list(results)

        for doc, result in zip(docs_list, results_list):
            # Create a new result with the same structure as the original
            entities_with_position: list[Entity] = []

            # Get the original text from the document
            doc_text = doc.text or ""

            if hasattr(result, "entities"):
                # Process entities from result if available
                entities_with_position = self._find_entity_positions(doc_text, result, entities_with_position)
                # Create a new result with the updated entities
                new_result = Entities(text=doc_text, entities=entities_with_position)
                doc.results[self._task_id] = new_result
            else:
                # Default empty result
                doc.results[self._task_id] = Entities(text=doc_text, entities=[])

        return docs_list


class DSPyNER(NERBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    @property
    def _prompt_template(self) -> str | None:
        return None

    @property
    def _prompt_signature_description(self) -> str | None:
        return """
        A named entity recognition result that represents named entities from the provided text. For each entity found it includes:
        - exact text of the entity
        - a context string that contains the exact entity text along with a few surrounding words 
          (two or three surronding words). The context includes the entity text itself.
        - if the same entity appears multiple times in the text, each occurrence is listed separately with its 
        own context
        - the entity type from the provided list of entity types. Only entities of the specified types are included.
        """

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

        Prediction.__doc__ = jinja2.Template(self.prompt_signature_description).render()

        return Prediction

    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return dspy_.InferenceMode.predict

    def consolidate(
        self, results: Iterable[dspy_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[dspy_.Result]:
        results = list(results)
        # Process each document (which may consist of multiple chunks)
        for doc_offset in docs_offsets:
            doc_results = results[doc_offset[0] : doc_offset[1]]

            # Combine all entities from all chunks
            all_entities: list[Entity] = []

            # Process each chunk for this document
            for chunk_result in doc_results:
                if not hasattr(chunk_result, "entities") or not chunk_result.entities:
                    continue

                # Process entities in this chunk
                for entity in chunk_result.entities:
                    all_entities.append(entity)

            # Create a consolidated result for this document
            yield dspy.Prediction.from_completions({"entities": [all_entities]}, signature=self.prompt_signature)


class PydanticBasedNER(NERBridge[pydantic.BaseModel, pydantic.BaseModel, EngineInferenceMode]):
    @property
    def _prompt_template(self) -> str | None:
        return """
        Your goal is to extract named entities from the text. Only extract entities of the specified types: 
        {{ entity_types }}.

        {% if examples|length > 0 -%}
            Examples:
            ----------
            {%- for example in examples %}
                Text: "{{ example.text }}":
                Entities: [
                    {%- for entity in example.entities %}
                    {
                        "text": "{{ entity.text }}",
                        
                        "entity_type": "{{ entity.entity_type }}"
                    }{%- if not loop.last %}, {% endif %}
                    {%- endfor %}
                ]
            {% endfor -%}
            ----------
        {% endif %}

        For each entity:
        - Extract the exact text of the entity
        - Include a SHORT context string that contains ONLY the entity and AT MOST 3 words before and 3 words after it. 
          DO NOT include the entire text as context. DO NOT include words that are not present in the original text 
          as introductory words (Eg. 'Text:' before context string).
        - Specify which type of entity it is (must be one of the provided entity types)

        IMPORTANT:
        - If the same entity appears multiple times in the text, extract each occurrence separately with its own context

        Text: {{ text }}
        Entity Types: {{ entity_types }}
        """

    @property
    def _prompt_signature_description(self) -> str | None:
        return """
        A named entity recognition signature that extracts named entities from the provided text. For each entity found:
        - it extracts the exact text of the entity
        - it includes a context string that contains ONLY the entity and AT MOST 3 words before and 3 words 
          after it. DO NOT include the entire text as context.
        - it specifies the entity type from the provided list of entity types
        Only extract entities of the specified types.
        """

    @cached_property
    def _prompt_signature(self) -> type[pydantic.BaseModel]:
        entity_types = self._entities
        LiteralType = Literal[*entity_types]  # type: ignore[valid-type]

        class EntityWithContext(pydantic.BaseModel):
            text: str
            context: str
            entity_type: LiteralType

        class Prediction(pydantic.BaseModel):
            entities: list[EntityWithContext] = []

        if self.prompt_signature_description:
            Prediction.__doc__ = jinja2.Template(self.prompt_signature_description).render()

        return Prediction

    def consolidate(
        self, results: Iterable[pydantic.BaseModel], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[pydantic.BaseModel]:
        results = list(results)
        prediction_class = self._prompt_signature  # Get the class, not an instance

        # Process each document (which may consist of multiple chunks)
        for doc_offset in docs_offsets:
            doc_results = results[doc_offset[0] : doc_offset[1]]

            # Combine all entities from all chunks
            all_entities: list[dict[str, Any]] = []

            # Process each chunk for this document
            for chunk_result in doc_results:
                if not hasattr(chunk_result, "entities") or not chunk_result.entities:
                    continue

                # Process entities in this chunk
                for entity in chunk_result.entities:
                    # We just need to combine all entities from all chunks
                    all_entities.append(entity)

            # Create a consolidated result for this document - instantiate the class with entities
            yield prediction_class(entities=all_entities)


class OutlinesNER(PydanticBasedNER[outlines_.InferenceMode]):
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return outlines_.InferenceMode.json

    @cached_property
    def prompt_signature(self) -> type[pydantic.BaseModel]:
        return self._prompt_signature


class OllamaNER(PydanticBasedNER[ollama_.InferenceMode]):
    @property
    def inference_mode(self) -> ollama_.InferenceMode:
        return ollama_.InferenceMode.chat

    @property
    def prompt_signature(self) -> type[pydantic.BaseModel]:
        return self._prompt_signature


class LangChainNER(PydanticBasedNER[langchain_.InferenceMode]):
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return langchain_.InferenceMode.structured_output

    @property
    def prompt_signature(self) -> type[pydantic.BaseModel]:
        return self._prompt_signature


class InstructorNER(PydanticBasedNER[instructor_.InferenceMode]):
    @property
    def inference_mode(self) -> instructor_.InferenceMode:
        return instructor_.InferenceMode.chat

    @property
    def prompt_signature(self) -> type[pydantic.BaseModel]:
        return self._prompt_signature


class GliXNER(NERBridge[list[str], glix_.Result, glix_.InferenceMode]):
    def __init__(
        self,
        entities: list[str],
        task_id: str,
        prompt_template: str | None,
        prompt_signature_desc: str | None,
    ):
        """
        Initializes GliXNER bridge.
        :param entities: List of entity types to extract.
        :param task_id: Task ID.
        :param prompt_template: Custom prompt template.
        :param prompt_signature_desc: Custom prompt signature description.
        """
        super().__init__(
            entities=entities,
            task_id=task_id,
            prompt_template=prompt_template,
            prompt_signature_desc=prompt_signature_desc,
        )

    @property
    def _prompt_template(self) -> str | None:
        return None

    @property
    def _prompt_signature_description(self) -> str | None:
        return None

    @property
    def prompt_signature(self) -> list[str]:
        return self._entities

    @property
    def inference_mode(self) -> glix_.InferenceMode:
        return glix_.InferenceMode.ner

    def consolidate(
        self, results: Iterable[glix_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[glix_.Result]:
        results = list(results)

        # Simply group results by document without trying to adjust positions
        # Position adjustment will happen in the integrate function
        for doc_offset in docs_offsets:
            doc_results = results[doc_offset[0] : doc_offset[1]]
            all_entities: list[dict[str, Any]] = []

            # Keep track of which chunk each entity came from
            for chunk_idx, chunk_result in enumerate(doc_results):
                # Process entities in this chunk
                for entity in chunk_result:
                    if isinstance(entity, dict):
                        # Add chunk index to the entity for reference in integrate
                        entity_copy = entity.copy()
                        entity_copy["chunk_idx"] = chunk_idx
                        all_entities.append(entity_copy)

            # Yield results for this document (flattened list of entities)
            yield all_entities

    def integrate(self, results: Iterable[glix_.Result], docs: Iterable[Doc]) -> Iterable[Doc]:
        docs_list = list(docs)
        results_list = list(results)

        class Entity(pydantic.BaseModel):
            text: str
            start: int
            end: int
            entity_type: str

        class Entities(pydantic.BaseModel):
            text: str
            entities: list[Entity] = []

        # Process each document
        for doc, result in zip(docs_list, results_list):
            entities_list: list[Entity] = []
            doc_text = doc.text if doc.text is not None else ""

            # Get chunk information from the document
            chunk_offsets = []
            if hasattr(doc, "chunks") and doc.chunks:
                # Calculate beginning position of each chunk in the original text
                current_offset = 0
                for chunk in doc.chunks:
                    chunk_offsets.append(current_offset)
                    current_offset += len(chunk)

            # Process entities in this document
            if result:
                for entity_dict in result:
                    if not isinstance(entity_dict, dict):
                        continue

                    try:
                        entity_text = str(entity_dict.get("text", ""))
                        entity_start = int(entity_dict.get("start", 0))
                        entity_end = int(entity_dict.get("end", 0))
                        entity_type = str(entity_dict.get("label", ""))

                        # Get the chunk index (added in consolidate)
                        chunk_idx = int(entity_dict.get("chunk_idx", 0))

                        # Add chunk offset to entity positions
                        adjusted_start = entity_start
                        adjusted_end = entity_end

                        if chunk_offsets and chunk_idx < len(chunk_offsets):
                            # Adjust positions based on chunk offset
                            adjusted_start += chunk_offsets[chunk_idx]
                            adjusted_end += chunk_offsets[chunk_idx]

                        entities_list.append(
                            Entity(
                                text=entity_text,
                                start=adjusted_start,
                                end=adjusted_end,
                                entity_type=entity_type,
                            )
                        )
                    except (ValueError, TypeError) as e:
                        print(f"Error processing entity: {e}")
                        continue

            # Create the final entities object and store in document results
            entities_obj = Entities(text=doc_text, entities=entities_list)
            doc.results[self._task_id] = entities_obj

        return docs_list
