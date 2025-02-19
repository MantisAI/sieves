import abc
from collections.abc import Iterable
from functools import cached_property
from typing import TypeVar

import dspy
import jinja2
import pydantic

from sieves.data import Doc
from sieves.engines import EngineInferenceMode, dspy_, instructor_, langchain_, ollama_, outlines_, glix_, huggingface_
from sieves.tasks.predictive.bridges import Bridge

_BridgePromptSignature = TypeVar("_BridgePromptSignature")
_BridgeResult = TypeVar("_BridgeResult")

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
        super().__init__(task_id=task_id, prompt_template=prompt_template, prompt_signature_desc=prompt_signature_desc)
        self._entities = entities


class DSPyNER(NERBridge[dspy_.PromptSignature, dspy_.Result, dspy_.InferenceMode]):
    @property
    def _prompt_template(self) -> str | None:
        return None

    @property
    def _prompt_signature_description(self) -> str | None:
        return """
        Extract named entities from the provided text. For each entity found:
        - Extract the exact text of the entity
        - Provide the starting and ending character positions
        - Specify the entity type from the provided list of entity types
        Only extract entities of the specified types.
        """

    @cached_property
    def prompt_signature(self) -> type[dspy_.PromptSignature]:
        class Entity(dspy.Signature):
            text: str = dspy.OutputField(description="The extracted entity text")
            start: int = dspy.OutputField(description="Starting character position of the entity")
            end: int = dspy.OutputField(description="Ending character position of the entity")
            entity: str = dspy.OutputField(description="The type of entity")
        
        class Entities(dspy.Signature):
            text: str = dspy.InputField(description="Text to extract entities from")
            entity_types: list[str] = dspy.InputField(default_factory=lambda: self._entities, description="List of entity types to extract")
            entities: list[Entity] = dspy.OutputField(description="List of entities found in the text")

        Entities.__doc__ = jinja2.Template(self.prompt_signature_description).render()

        return Entities

    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return dspy_.InferenceMode.chain_of_thought
    
    def integrate(self, results: Iterable[dspy_.Result], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            doc.results[self._task_id] = result.entities
        return docs
    
    
class HuggingFaceNER(NERBridge[list[str], huggingface_.Result, huggingface_.InferenceMode]):
    @property
    def _prompt_template(self) -> str | None:
        return """
        TO DEFINE
        """
    
    @property
    def _prompt_signature_description(self) -> str | None:
        return None
    
    @property
    def prompt_signature(self) -> list[str]:
        return self._entities
    
    @property
    def inference_mode(self) -> huggingface_.InferenceMode:
        return huggingface_.InferenceMode.complete
    
    def integrate(self, results: Iterable[huggingface_.Result], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            doc.results[self._task_id] = result.entities
        return docs
    
    
class PydanticBasedNER(NERBridge[pydantic.BaseModel, pydantic.BaseModel, outlines_.InferenceMode]):
    @property
    def _prompt_template(self) -> str | None:
        return """
        TO DEFINE
        """
    @property
    def _prompt_signature_description(self) -> str | None:
        return None
    @cached_property
    def _prompt_signature(self) -> type[pydantic.BaseModel]:
        prompt_sig = pydantic.create_model(
            "Entities",
            __base__=pydantic.BaseModel,
            # reasoning=(str, ...),
            **{entity: (str, ...) for entity in self._entities}
        )
    
        if self.prompt_signature_description:
            prompt_sig.__doc__ = jinja2.Template(self.prompt_signature_description).render()
        # ASSERTION
        return prompt_sig
    def integrate(self, results: Iterable[pydantic.BaseModel], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            doc.results[self._task_id] = result.entities
        return docs
    

class OutlinesNER(PydanticBasedNER[outlines_.InferenceMode]):
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return outlines_.InferenceMode.json


class OllamaNER(PydanticBasedNER[ollama_.InferenceMode]):
    @property
    def inference_mode(self) -> ollama_.InferenceMode:
        return ollama_.InferenceMode.chat


class LangChainNER(PydanticBasedNER[langchain_.InferenceMode]):
    @property
    def inference_mode(self) -> langchain_.InferenceMode:
        return langchain_.InferenceMode.structured_output


class InstructorNER(PydanticBasedNER[instructor_.InferenceMode]):
    @property
    def inference_mode(self) -> instructor_.InferenceMode:
        return instructor_.InferenceMode.chat

