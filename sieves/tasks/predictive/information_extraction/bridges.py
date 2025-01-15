import abc
from collections.abc import Iterable
from functools import cached_property
from typing import Any, TypeVar

import dspy
import jinja2
import pydantic

from sieves.data import Doc
from sieves.engines import dspy_, ollama_, outlines_
from sieves.tasks.core import Bridge

BridgePromptSignature = TypeVar("BridgePromptSignature", covariant=True)
BridgeInferenceMode = TypeVar("BridgeInferenceMode", covariant=True)
BridgeResult = TypeVar("BridgeResult")


class InformationExtractionBridge(
    Bridge[BridgePromptSignature, BridgeInferenceMode, BridgeResult],
    abc.ABC,
):
    def __init__(
        self,
        task_id: str,
        custom_prompt_template: str | None,
        entity_type: type[pydantic.BaseModel],
    ):
        """
        Initializes InformationExtractionBridge.
        :param task_id: Task ID.
        :param custom_prompt_template: Custom prompt template.
        :param entity_type: Type to extract.
        """
        super().__init__(task_id, custom_prompt_template)
        self._entity_type = entity_type


class DSPyInformationExtraction(InformationExtractionBridge[dspy_.PromptSignature, dspy_.InferenceMode, dspy_.Result]):
    @property
    def prompt_template(self) -> str | None:
        return (
            self._custom_prompt_template
            or """
            Find all occurences of this kind of entitity within the text.
            """
        )

    @cached_property
    def prompt_signature(self) -> type[dspy_.PromptSignature]:  # type: ignore[valid-type]
        extraction_type = self._entity_type

        class Entities(dspy.Signature):  # type: ignore[misc]
            text: str = dspy.InputField()
            entities: list[extraction_type] = dspy.OutputField()  # type: ignore[valid-type]

        Entities.__doc__ = jinja2.Template(self.prompt_template).render()

        return Entities

    @property
    def inference_mode(self) -> dspy_.InferenceMode:
        return dspy_.InferenceMode.predict

    def extract(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        return ({"text": doc.text if doc.text else None} for doc in docs)

    def integrate(self, results: Iterable[dspy_.Result], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            assert len(result.completions.entities) == 1
            doc.results[self._task_id] = result.completions.entities[0]
        return docs

    def consolidate(
        self, results: Iterable[dspy_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[dspy_.Result]:
        results = list(results)
        entity_type = self._entity_type
        entity_type_is_frozen = entity_type.model_config.get("frozen", False)

        # Merge all found entities.
        for doc_offset in docs_offsets:
            entities: list[entity_type] = []  # type: ignore[valid-type]
            seen_entities: set[entity_type] = set()  # type: ignore[valid-type]

            for res in results[doc_offset[0] : doc_offset[1]]:
                assert len(res.completions.entities) == 1
                if entity_type_is_frozen:
                    # Ensure not to add duplicate entities.
                    for entity in res.completions.entities[0]:
                        if entity not in seen_entities:
                            entities.append(entity)
                            seen_entities.add(entity)
                else:
                    entities.extend(res.completions.entities[0])

            yield dspy.Prediction.from_completions(
                {"entities": [entities]},
                signature=self.prompt_signature,
            )


class PydanticBasedInformationExtraction(
    InformationExtractionBridge[
        type[pydantic.BaseModel], outlines_.InferenceMode | ollama_.InferenceMode, pydantic.BaseModel
    ],
    abc.ABC,
):
    @property
    def prompt_template(self) -> str | None:
        # {% for key, value in entity.model_dump().items() %}{{ key }}: {{ value }}{% endfor %}
        return (
            self._custom_prompt_template
            or """
            Find all occurences of this kind of entitity within the text.

            {% if examples|length > 0 -%}
                Examples:
                ----------
                {%- for example in examples %}
                    Text: "{{ example.text }}":
                    Output: {{ example.entities }}
                {% endfor -%}
                ----------
            {% endif -%}

            ========
            Text: {{ text }}
            Output: 
            """
        )

    @cached_property
    def prompt_signature(self) -> type[pydantic.BaseModel]:
        entity_type = self._entity_type

        class Entity(pydantic.BaseModel, frozen=True):
            entities: list[entity_type]  # type: ignore[valid-type]

        return Entity

    def integrate(self, results: Iterable[pydantic.BaseModel], docs: Iterable[Doc]) -> Iterable[Doc]:
        for doc, result in zip(docs, results):
            assert hasattr(result, "entities")
            doc.results[self._task_id] = result.entities
        return docs

    def consolidate(
        self, results: Iterable[pydantic.BaseModel], docs_offsets: list[tuple[int, int]]
    ) -> Iterable[pydantic.BaseModel]:
        results = list(results)
        entity_type = self._entity_type
        entity_type_is_frozen = entity_type.model_config.get("frozen", False)

        # Determine label scores for chunks per document.
        for doc_offset in docs_offsets:
            entities: list[entity_type] = []  # type: ignore[valid-type]
            seen_entities: set[entity_type] = set()  # type: ignore[valid-type]

            for res in results[doc_offset[0] : doc_offset[1]]:
                assert hasattr(res, "entities")
                if entity_type_is_frozen:
                    # Ensure not to add duplicate entities.
                    for entity in res.entities:
                        if entity not in seen_entities:
                            entities.append(entity)
                            seen_entities.add(entity)
                else:
                    entities.extend(res.entities)

            yield self.prompt_signature(entities=entities)

    def extract(self, docs: Iterable[Doc]) -> Iterable[dict[str, Any]]:
        return ({"text": doc.text if doc.text else None} for doc in docs)


class OutlinesInformationExtraction(PydanticBasedInformationExtraction):
    @property
    def inference_mode(self) -> outlines_.InferenceMode:
        return outlines_.InferenceMode.json


class OllamaInformationExtraction(PydanticBasedInformationExtraction):
    @property
    def inference_mode(self) -> ollama_.InferenceMode:
        return ollama_.InferenceMode.chat
