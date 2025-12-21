"""Bridge base class and types."""

from __future__ import annotations

import abc
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal, TypeVar, override

import gliner2
import pydantic

from sieves.data import Doc
from sieves.model_wrappers import ModelWrapperInferenceMode, gliner_
from sieves.model_wrappers.types import ModelSettings

TaskPromptSignature = TypeVar("TaskPromptSignature", covariant=True)
TaskResult = TypeVar("TaskResult")
TaskBridge = TypeVar("TaskBridge", bound="Bridge[TaskPromptSignature, TaskResult, ModelWrapperInferenceMode]")  # type: ignore[valid-type]


class EntityWithContext(pydantic.BaseModel):
    """Entity mention with text span and type."""

    text: str
    context: str
    entity_type: str


class Entity(pydantic.BaseModel):
    """Class for storing entity information."""

    text: str
    start: int
    end: int
    entity_type: str

    def __eq__(self, other: object) -> bool:
        """Compare two entities.

        :param other: Other entity to compare with.
        :return: True if entities are equal, False otherwise.
        """
        if not isinstance(other, Entity):
            return False
        # Two entities are equal if they have the same start, end, text and entity_type
        return (
            self.start == other.start
            and self.end == other.end
            and self.text == other.text
            and self.entity_type == other.entity_type
        )

    def __hash__(self) -> int:
        """Compute entity hash.

        :returns: Entity hash.
        """
        return hash((self.start, self.end, self.text, self.entity_type))


class Entities(pydantic.BaseModel):
    """Collection of entities with associated text."""

    entities: list[Entity]
    text: str


class Bridge[TaskPromptSignature, TaskResult, ModelWrapperInferenceMode](abc.ABC):
    """Bridge base class."""

    def __init__(self, task_id: str, prompt_instructions: str | None, overwrite: bool, model_settings: ModelSettings):
        """Initialize new bridge.

        :param task_id: Task ID.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param overwrite: Whether to overwrite text with produced text. Considered only by bridges for tasks producing
            fluent text - like translation, summarization, PII masking, etc.
        :param model_settings: Model settings including inference_mode.
        """
        self._task_id = task_id
        self._custom_prompt_instructions = prompt_instructions
        self._overwrite = overwrite
        self._model_settings = model_settings

    @property
    @abc.abstractmethod
    def _default_prompt_instructions(self) -> str:
        """Return default prompt instructions.

        Instructions are injected at the beginning of each prompt.

        :return: Default prompt instructions.
        """

    @property
    def _prompt_instructions(self) -> str:
        """Returns prompt instructions.

        :returns: If `_custom_prompt_instructions` is set, this is used. Otherwise, `_default_prompt_instructions` is
            used.
        """
        return self._custom_prompt_instructions or self._default_prompt_instructions

    @property
    @abc.abstractmethod
    def _prompt_example_template(self) -> str | None:
        """Return default prompt template for example injection.

        Examples are injected between instructions and conclusions.

        :return: Default prompt example template.
        """

    @property
    @abc.abstractmethod
    def _prompt_conclusion(self) -> str | None:
        """Return prompt conclusion.

        Prompt conclusions are injected at the end of each prompt.

        :return: Default prompt conclusion.
        """

    @property
    def prompt_template(self) -> str:
        """Return prompt template.

        Chains `_prompt_instructions`, `_prompt_example_template` and `_prompt_conclusion`.

        Note: different model have different expectations as to how a prompt should look like. E.g. outlines supports
        the Jinja 2 templating format for insertion of values and few-shot examples, whereas DSPy integrates these
        things in a different value in the workflow and hence expects the prompt not to include these things. Mind
        model-specific expectations when creating a prompt template.
        :return str | None: Prompt template as string. None if not used by model wrapper.
        """
        return f"""
        {self._custom_prompt_instructions or self._prompt_instructions}
        {self._prompt_example_template or ""}
        {self._prompt_conclusion or ""}
        """

    @property
    @abc.abstractmethod
    def prompt_signature(self) -> type[TaskPromptSignature] | TaskPromptSignature:
        """Create output signature.

        E.g.: `Signature` in DSPy, Pydantic objects in outlines, JSON schema in jsonformers.
        This is model type-specific.

        :return type[_TaskPromptSignature] | _TaskPromptSignature: Output signature object. This can be an instance
            (e.g. a regex string) or a class (e.g. a Pydantic class).
        """

    @property
    @abc.abstractmethod
    def inference_mode(self) -> ModelWrapperInferenceMode:
        """Return inference mode.

        :return ModelWrapperInferenceMode: Inference mode.
        """

    def extract(self, docs: Sequence[Doc]) -> Sequence[dict[str, Any]]:
        """Extract all values from doc instances that are to be injected into the prompts.

        :param docs: Docs to extract values from.
        :return: All values from doc instances that are to be injected into the prompts as a sequence.
        """
        return [{"text": doc.text if doc.text else None} for doc in docs]

    @abc.abstractmethod
    def integrate(self, results: Sequence[TaskResult], docs: list[Doc]) -> list[Doc]:
        """Integrate results into Doc instances.

        :param results: Results from prompt executable.
        :param docs: Doc instances to update.
        :return: Updated doc instances as a list.
        """

    @abc.abstractmethod
    def consolidate(self, results: Sequence[TaskResult], docs_offsets: list[tuple[int, int]]) -> Sequence[TaskResult]:
        """Consolidate results for document chunks into document results.

        :param results: Results per document chunk.
        :param docs_offsets: Chunk offsets per document. Chunks per document can be obtained with
            `results[docs_chunk_offsets[i][0]:docs_chunk_offsets[i][1]]`.
        :return: Results per document as a sequence.
        """


class GliNERBridge(Bridge[gliner2.inference.engine.Schema, gliner_.Result, gliner_.InferenceMode]):
    """Bridge for GLiNER2 models."""

    def __init__(
        self,
        task_id: str,
        prompt_instructions: str | None,
        prompt_signature: gliner2.inference.engine.Schema | gliner2.inference.engine.StructureBuilder,
        model_settings: ModelSettings,
        inference_mode: gliner_.InferenceMode,
        mode: Literal["multi", "single"] = "multi",
    ):
        """Initialize GLiNER2 bridge.

        Important: currently only GLiNER2 schemas/structures with one key each are supported. We do NOT support
        composite requests like `create_schema().entities().classification(). ...`.

        :param task_id: Task ID.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param prompt_signature: GLiNER2 schema (list of field definitions).
        :param model_settings: Model settings including inference_mode.
        :param mode: Extraction mode. If "multi", all occurrences of the entity are extracted. If "single", exactly one
            (or no) entity is extracted. Only used if `inference_mode == InferenceMode.structure`, ignored otherwise.
        """
        super().__init__(
            task_id=task_id,
            prompt_instructions=prompt_instructions,
            overwrite=False,
            model_settings=model_settings,
        )
        self._prompt_signature = prompt_signature
        # If prompt signature is a structure, we create a Pydantic representation of it for easier downstream result
        # processing - e.g. when creating a HF dataset.
        self._prompt_signature_pydantic = (
            self.schema_to_pydantic()
            if isinstance(prompt_signature, gliner2.inference.engine.StructureBuilder)
            else None
        )

        self._inference_mode = inference_mode
        self._mode = mode

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        # GLiNER2 doesn't support custom instructions.
        return ""

    @override
    @property
    def _prompt_example_template(self) -> str | None:
        return None

    @override
    @property
    def _prompt_conclusion(self) -> str | None:
        return None

    @override
    @property
    def prompt_signature(self) -> gliner2.inference.engine.Schema | gliner2.inference.engine.StructureBuilder:
        return self._prompt_signature

    @property
    def prompt_signature_pydantic(self) -> type[pydantic.BaseModel] | None:
        """Return Pydantic model representation of GLiNER2 schema.

        Returns:
            Pydantic model representation of GLiNER2 schema.
        """
        return self._prompt_signature_pydantic

    @override
    @property
    def inference_mode(self) -> gliner_.InferenceMode:
        return self._model_settings.inference_mode or self._inference_mode

    def schema_to_pydantic(self) -> type[pydantic.BaseModel]:
        """Convert a Gliner2 Schema object to Pydantic models.

        If the schema is a structure with more than one entry, a wrapper class `Schema` is created.

        Returns:
            Pydantic model representation of GLiNER2 schema.
        """
        if isinstance(self._prompt_signature, gliner2.inference.engine.StructureBuilder):
            field_metadata = self._prompt_signature.schema._field_metadata
        else:
            assert isinstance(self._prompt_signature, gliner2.inference.engine.Schema)
            field_metadata = self._prompt_signature._field_metadata

        # Group fields by class name.
        classes: dict[str, dict[str, Any]] = {}
        for key, meta in field_metadata.items():
            class_name, field_name = key.split(".")
            if class_name not in classes:
                classes[class_name] = {}
            classes[class_name][field_name] = meta

        # Create models for each class
        models: dict[str, type[pydantic.BaseModel]] = {}
        for class_name, fields in classes.items():
            field_definitions = {}
            for field_name, meta in fields.items():
                dtype = meta["dtype"]
                choices = meta["choices"]

                # Determine the field type.
                inner_field_type = Literal[*choices] if choices else str  # type: ignore[invalid-type-form]
                field_type = list[inner_field_type] if dtype == "list" else inner_field_type
                field_definitions[field_name] = (field_type, ...)

            model = pydantic.create_model(class_name, **field_definitions)
            models[class_name] = model

        # Create wrapper "Schema" model with lowercase attribute names if more than one structure is present.
        if len(models) > 1:
            raise TypeError(
                "Composite GliNER2 schemas are not supported. Use a single structure/entitity/classification per Sieves"
                " task."
            )

        return models[list(models.keys())[0]]

    @override
    def integrate(self, results: Sequence[gliner_.Result], docs: list[Doc]) -> list[Doc]:
        for doc, result in zip(docs, results):
            match self._inference_mode:
                # Used by: Classification
                case gliner_.InferenceMode.classification:
                    assert hasattr(self._prompt_signature.schema, "__getitem__")
                    is_multilabel = self._prompt_signature.schema["classifications"][0]["multi_label"]

                    if is_multilabel:
                        doc.results[self._task_id] = []
                        for res in sorted(result, key=lambda x: x["score"], reverse=True):
                            assert isinstance(res, dict)
                            doc.results[self._task_id].append((res["label"], res["score"]))

                    else:
                        doc.results[self._task_id] = (result[0]["label"], result[0]["score"])

                # Used by: NER
                case gliner_.InferenceMode.entities:
                    if isinstance(result, list):
                        # Result is a list of entity dictionaries.
                        doc.results[self._task_id] = Entities(
                            entities=[
                                Entity.model_validate({k: v for k, v in res.items() if k != "confidence"})
                                for res in result
                            ],
                            text=doc.text or "",
                        )
                    else:
                        doc.results[self._task_id] = result

                # Used by: InformationExtraction
                case gliner_.InferenceMode.structure:
                    entity_type_name = list(result.keys())[0]
                    assert issubclass(self._prompt_signature_pydantic, pydantic.BaseModel)

                    extracted_entities = [
                        self._prompt_signature_pydantic.model_validate(
                            {key: value["text"] for key, value in entity.items()}
                        )
                        for entity in result[entity_type_name]
                    ]

                    # This covers single/multi mode for information extraction.
                    if self._mode == "multi":
                        doc.results[self._task_id] = extracted_entities
                    else:
                        doc.results[self._task_id] = extracted_entities[0] if extracted_entities else None

        return docs

    @override
    def consolidate(
        self, results: Sequence[gliner_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[gliner_.Result]:
        consolidated_results: list[gliner_.Result] = []

        # Determine label scores for chunks per document.
        for doc_offset in docs_offsets:
            scores: dict[str, float] = defaultdict(lambda: 0)
            entities: dict[str, list[str] | dict[str, str | list[str]]] = {}
            all_entities_list: list[dict[str, Any]] = []

            # In single mode for structure, we want to keep track of the highest confidence entity.
            highest_confidence_entity: dict[str, Any] | None = None
            max_confidence: float = -1.0

            for res in results[doc_offset[0] : doc_offset[1]]:
                match self._inference_mode:
                    case gliner_.InferenceMode.classification:
                        keys = list(res.keys())
                        assert len(keys) == 1, "Composite GliNER2 schemas are not supported."
                        extracted_res = res[keys[0]]

                        # In case of single-label: pad to list so that we can process in a unified way.
                        if isinstance(extracted_res, dict):
                            extracted_res = [extracted_res]

                        for entry in extracted_res:
                            # GliNER might use two different structures here, depending on the version.
                            if "label" in entry:
                                scores[entry["label"]] += entry["confidence"]
                            else:
                                keys = list(entry.keys())
                                assert len(keys) == 1, "Composite GliNER2 schemas are not supported."
                                for label, confidence in entry[keys[0]]:
                                    scores[label] += confidence

                    case gliner_.InferenceMode.entities:
                        for entity_type in res["entities"]:
                            items = res["entities"][entity_type]
                            if items:
                                if isinstance(items[0], dict):
                                    # Flatten into a list of entities (dicts).
                                    for item in items:
                                        all_entities_list.append({"entity_type": entity_type, **item})
                                else:
                                    if entity_type not in entities:
                                        entities[entity_type] = []
                                    relevant_entities: list[str] = entities[entity_type]  # type: ignore[assignment]
                                    relevant_entities.extend(items)

                    case gliner_.InferenceMode.structure:
                        for entity_type in res:
                            if entity_type not in entities:
                                entities[entity_type] = []
                            relevant_entities_struct: list[Any] = entities[entity_type]  # type: ignore[assignment]
                            relevant_entities_struct.extend(res[entity_type])

                            if self._mode == "single":
                                for entity in res[entity_type]:
                                    # Calculate average confidence for the entity structure.
                                    confidences = [
                                        field_val["confidence"]
                                        for field_val in entity.values()
                                        if "confidence" in field_val
                                    ]
                                    if confidences:
                                        avg_conf = sum(confidences) / len(confidences)
                                        if avg_conf > max_confidence:
                                            max_confidence = avg_conf
                                            highest_confidence_entity = {entity_type: [entity]}

            match self._inference_mode:
                case gliner_.InferenceMode.classification:
                    # Ensure that all labels have been assigned - GLiNER2 is sometimes negligent about this.
                    assert hasattr(self._prompt_signature.schema, "__getitem__")
                    for label in self._prompt_signature.schema["classifications"][0]["labels"]:
                        if label not in scores:
                            scores[label] = 0.0

                    # Average score, sort in descending order.
                    sorted_scores: list[dict[str, str | float]] = sorted(
                        (
                            {"label": attr, "score": score / (doc_offset[1] - doc_offset[0])}
                            for attr, score in scores.items()
                        ),
                        key=lambda x: x["score"],
                        reverse=True,
                    )

                    consolidated_results.append(sorted_scores)

                case gliner_.InferenceMode.entities | gliner_.InferenceMode.structure:
                    if self._inference_mode == gliner_.InferenceMode.structure and self._mode == "single":
                        consolidated_results.append(highest_confidence_entity or {list(res.keys())[0]: []})
                    elif all_entities_list:
                        consolidated_results.append(all_entities_list)
                    else:
                        consolidated_results.append(entities)

        return consolidated_results
