"""Bridges for NER task."""

import abc
import re
from collections.abc import Sequence
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
from sieves.tasks.predictive.schemas.ner import Entity, Result
from sieves.tasks.predictive.utils import convert_to_signature

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")


class NERBridge(Bridge[_BridgePromptSignature, _BridgeResult, ModelWrapperInferenceMode], abc.ABC):
    """Abstract base class for NER bridges."""

    def __init__(
        self,
        task_id: str,
        prompt_instructions: str | None,
        entities: list[str] | dict[str, str],
        model_settings: ModelSettings,
        prompt_signature: type[pydantic.BaseModel],
        model_type: ModelType,
        fewshot_examples: Sequence[pydantic.BaseModel] = (),
    ):
        """Initialize NER bridge.

        :param task_id: Task ID.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param entities: List of entities to extract or dict mapping labels to descriptions.
        :param model_settings: Settings for structured generation.
        :param prompt_signature: Unified Pydantic prompt signature.
        :param model_type: Model type.
        :param fewshot_examples: Few-shot examples.
        """
        assert model_type in {ModelType.dspy, ModelType.outlines, ModelType.langchain, ModelType.gliner}

        super().__init__(
            task_id=task_id,
            prompt_instructions=prompt_instructions,
            overwrite=False,
            model_settings=model_settings,
            prompt_signature=prompt_signature,
            model_type=model_type,
            fewshot_examples=fewshot_examples,
        )
        if isinstance(entities, dict):
            self._entities = list(entities.keys())
            self._entity_descriptions = entities
        else:
            self._entities = list(entities)
            self._entity_descriptions = {}

    @override
    @property
    def prompt_signature(self) -> _BridgePromptSignature:
        return convert_to_signature(
            model_cls=self._pydantic_signature,
            model_type=self.model_type,
            mode="entities",
        )  # type: ignore[return-value]

    def _get_entity_descriptions(self) -> str:
        """Return a string with the entity descriptions.

        :return: A string with the entity descriptions.
        """
        entities_with_descriptions: list[str] = []
        for entity in self._entities:
            if entity in self._entity_descriptions:
                entities_with_descriptions.append(
                    f"  <entity_description>\n    <entity>{entity}</entity>\n    <description>"
                    f"{self._entity_descriptions[entity]}</description>\n  </entity_description>"
                )
            else:
                entities_with_descriptions.append(f"  <entity>{entity}</entity>")

        entity_desc_string = "\n".join(entities_with_descriptions)
        return f"<entity_descriptions>\n{entity_desc_string}\n</entity_descriptions>"

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
            score = getattr(entity_with_context, "score", None)

            if not entity_text:
                continue

            if context is None:
                new_entities.append(
                    Entity(
                        text=entity_text,
                        start=-1,
                        end=-1,
                        entity_type=entity_type,
                        score=score,
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
                        score=score,
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
                new_result = Result(text=doc_text, entities=entities_with_position)
                doc.results[self._task_id] = new_result
            else:
                # Default empty result
                doc.results[self._task_id] = Result(text=doc_text, entities=[])

        return docs_list


class DSPyNER(NERBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    """DSPy bridge for NER."""

    @override
    def _validate(self) -> None:
        assert self._model_type == ModelType.dspy

    @override
    @property
    def model_type(self) -> ModelType:
        return ModelType.dspy

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        return ""

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


class PydanticNER(NERBridge[pydantic.BaseModel, pydantic.BaseModel, ModelWrapperInferenceMode], abc.ABC):
    """Base class for Pydantic-based NER bridges."""

    @override
    def _validate(self) -> None:
        assert self._model_type in {ModelType.langchain, ModelType.outlines}

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        entity_info = self._get_entity_descriptions() if self._entity_descriptions else ""
        return (
            "Your goal is to extract named entities from the text. Only extract entities of the specified types:\n"
            f"{self._entities}.\n"
            f"{entity_info}\n\n"
            "For each entity:\n"
            "- Extract the exact text of the entity\n"
            "- Include a SHORT context string that contains ONLY the entity and AT MOST 3 words before and 3 words "
            "after it.\n"
            "  DO NOT include the entire text as context. DO NOT include words that are not present in the original "
            "text\n"
            "  as introductory words (Eg. 'Text:' before context string).\n"
            "- Specify which type of entity it is (must be one of the provided entity types)\n"
            "- Provide a confidence score between 0.0 and 1.0 for the extraction.\n\n"
            "IMPORTANT:\n"
            "- If the same entity appears multiple times in the text, extract each occurrence separately with its own "
            "context"
        )

    @override
    @property
    def _prompt_conclusion(self) -> str | None:
        return "===========\n\n<text>{{ text }}</text>\n<entity_types>{{ entity_types }}</entity_types>\n<entities>"

    @override
    def consolidate(
        self, results: Sequence[pydantic.BaseModel], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[pydantic.BaseModel]:
        assert issubclass(self.prompt_signature, pydantic.BaseModel)

        # Process each document (which may consist of multiple chunks).
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
