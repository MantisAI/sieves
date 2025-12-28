"""Bridge for GLiNER2 models."""

from __future__ import annotations

import types
import typing
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal, override

import gliner2
import pydantic

from sieves.data import Doc
from sieves.model_wrappers import ModelType, gliner_
from sieves.model_wrappers.types import ModelSettings
from sieves.tasks.predictive.bridges import Bridge
from sieves.tasks.predictive.schemas.classification import ResultMultiLabel, ResultSingleLabel
from sieves.tasks.predictive.schemas.information_extraction import (
    ResultMulti as IEResultMulti,
)
from sieves.tasks.predictive.schemas.information_extraction import (
    ResultSingle as IEResultSingle,
)
from sieves.tasks.predictive.schemas.ner import Entity as NEREntity
from sieves.tasks.predictive.schemas.ner import Result as NERResult
from sieves.tasks.predictive.schemas.relation_extraction import (
    RelationEntity as RERelationEntity,
)
from sieves.tasks.predictive.schemas.relation_extraction import (
    RelationTriplet as RERelationTriplet,
)
from sieves.tasks.predictive.schemas.relation_extraction import (
    Result as RERelationResult,
)
from sieves.tasks.predictive.utils import convert_to_signature


class GliNERBridge(Bridge[gliner2.inference.engine.Schema, gliner_.Result, gliner_.InferenceMode]):
    """Bridge for GLiNER2 models."""

    def __init__(
        self,
        task_id: str,
        prompt_instructions: str | None,
        prompt_signature: type[pydantic.BaseModel],
        model_settings: ModelSettings,
        inference_mode: gliner_.InferenceMode,
        mode: Literal["multi", "single"] = "multi",
    ):
        """Initialize GLiNER2 bridge.

        Important: currently only GLiNER2 schemas/structures with one key each are supported. We do NOT support
        composite requests like `create_schema().entities().classification(). ...`.

        :param task_id: Task ID.
        :param prompt_instructions: Custom prompt instructions. If None, default instructions are used.
        :param prompt_signature: Unified Pydantic prompt signature.
        :param model_settings: Model settings including inference_mode.
        :param mode: Extraction mode. If "multi", all occurrences of the entity are extracted. If "single", exactly one
            (or no) entity is extracted. Only used if `inference_mode == InferenceMode.structure`, ignored otherwise.
        """
        super().__init__(
            task_id=task_id,
            prompt_instructions=prompt_instructions,
            overwrite=False,
            model_settings=model_settings,
            prompt_signature=prompt_signature,
            model_type=ModelType.gliner,
        )
        self._inference_mode = inference_mode
        self._mode = mode

    @override
    @property
    def model_type(self) -> ModelType:
        return ModelType.gliner

    @override
    @property
    def prompt_signature(self) -> gliner2.inference.engine.Schema | gliner2.inference.engine.StructureBuilder:
        # Map internal inference mode to GliNER utility mode.
        mode_map = {
            gliner_.InferenceMode.classification: "classification",
            gliner_.InferenceMode.entities: "entities",
            gliner_.InferenceMode.structure: "structure",
            gliner_.InferenceMode.relations: "relations",
        }
        mode = mode_map[self.inference_mode]

        model_cls = self._pydantic_signature
        # Unwrap unified signature container if necessary.
        if mode == "structure" or mode == "entities":
            if "entities" in model_cls.model_fields:
                model_cls = model_cls.model_fields["entities"].annotation
                if typing.get_origin(model_cls) is list:
                    model_cls = typing.get_args(model_cls)[0]
            elif "entity" in model_cls.model_fields:
                model_cls = model_cls.model_fields["entity"].annotation
                # Handle Optional/Union.
                if typing.get_origin(model_cls) in (typing.Union, types.UnionType):
                    args = typing.get_args(model_cls)
                    model_cls = [t for t in args if t is not type(None)][0]

        elif mode == "relations":
            if "triplets" in model_cls.model_fields:
                model_cls = model_cls.model_fields["triplets"].annotation
                if typing.get_origin(model_cls) is list:
                    model_cls = typing.get_args(model_cls)[0]

        kwargs = {"mode": mode}
        if mode == "classification":
            kwargs["task"] = self._task_id

        prompt_signature = convert_to_signature(
            model_cls=model_cls,
            model_type=ModelType.gliner,
            **kwargs,
        )
        assert isinstance(prompt_signature, gliner2.inference.engine.Schema | gliner2.inference.engine.StructureBuilder)

        return prompt_signature

    @property
    def prompt_signature_pydantic(self) -> type[pydantic.BaseModel] | None:
        """Return Pydantic model representation of GLiNER2 schema.

        :return: Pydantic model representation of GLiNER2 schema.
        """
        return self._pydantic_signature

    @override
    @property
    def _default_prompt_instructions(self) -> str:
        # GLiNER2 doesn't support custom instructions.
        return ""

    @override
    @property
    def inference_mode(self) -> gliner_.InferenceMode:
        return self._model_settings.inference_mode or self._inference_mode

    @override
    def integrate(self, results: Sequence[gliner_.Result], docs: list[Doc]) -> list[Doc]:
        for doc, result in zip(docs, results):
            match self._inference_mode:
                # Used by: Classification
                case gliner_.InferenceMode.classification:
                    if self._mode == "multi":
                        label_scores: list[tuple[str, float]] = []
                        for res in sorted(result, key=lambda x: x["score"], reverse=True):
                            assert isinstance(res, dict)
                            label_scores.append((res["label"], res["score"]))
                        doc.results[self._task_id] = ResultMultiLabel(label_scores=label_scores)

                    else:
                        doc.results[self._task_id] = ResultSingleLabel(
                            label=result[0]["label"], score=result[0]["score"]
                        )

                # Used by: NER
                case gliner_.InferenceMode.entities:
                    if isinstance(result, list):
                        # Result is a list of entity dictionaries.
                        doc.results[self._task_id] = NERResult(
                            entities=[
                                NEREntity.model_validate(
                                    {k if k != "confidence" else "score": v for k, v in res.items()}
                                )
                                for res in result
                            ],
                            text=doc.text or "",
                        )
                    else:
                        doc.results[self._task_id] = result

                # Used by: InformationExtraction
                case gliner_.InferenceMode.structure:
                    if not result:
                        if self._mode == "multi":
                            doc.results[self._task_id] = IEResultMulti(entities=[])
                        else:
                            doc.results[self._task_id] = IEResultSingle(entity=None)
                        continue

                    entity_type_name = list(result.keys())[0]
                    # Unified signature container unwrapping
                    validation_model = self.prompt_signature_pydantic
                    if validation_model and "entities" in validation_model.model_fields:
                        # Multi-mode container
                        validation_model = validation_model.model_fields["entities"].annotation
                        if hasattr(validation_model, "__args__"):
                            validation_model = validation_model.__args__[0]

                    elif validation_model and "entity" in validation_model.model_fields:
                        # Single-mode container
                        validation_model = validation_model.model_fields["entity"].annotation
                        if hasattr(validation_model, "__args__"):
                            # Filter out None
                            validation_model = [t for t in validation_model.__args__ if t is not type(None)][0]

                    assert validation_model is not None and issubclass(validation_model, pydantic.BaseModel)

                    extracted_entities: list[pydantic.BaseModel] = []
                    for entity in result[entity_type_name]:
                        # Map fields and include score.
                        entity_data = {key: value["text"] for key, value in entity.items()}
                        # Calculate average confidence across all fields.
                        confidences = [v["confidence"] for v in entity.values() if "confidence" in v]
                        entity_data["score"] = sum(confidences) / len(confidences) if confidences else None

                        extracted_entities.append(validation_model.model_validate(entity_data))

                    # This covers single/multi mode for information extraction.
                    if self._mode == "multi":
                        doc.results[self._task_id] = IEResultMulti(entities=extracted_entities)
                    else:
                        doc.results[self._task_id] = IEResultSingle(
                            entity=extracted_entities[0] if extracted_entities else None
                        )

                # Used by: RelationExtraction
                case gliner_.InferenceMode.relations:
                    triplets: list[RERelationTriplet] = []
                    # GliNER2 relations output is a dict mapping task ids to relation types to lists of triplets:
                    # {'task_id': {'relation_type': [{'head': {...}, 'tail': {...}}, ...]}}  # noqa: ERA001
                    relation_data = result.get(self._task_id) or result.get("relation_extraction", {})
                    assert isinstance(relation_data, dict)

                    for rel_type, triplets_list in relation_data.items():
                        for triplet_data in triplets_list:
                            # Calculate score as average of head and tail confidence if they exist.
                            head_conf = triplet_data.get("head", {}).get("confidence")
                            tail_conf = triplet_data.get("tail", {}).get("confidence")
                            confidences = [c for c in (head_conf, tail_conf) if c is not None]
                            score = sum(confidences) / len(confidences) if confidences else None

                            triplets.append(
                                RERelationTriplet(
                                    head=RERelationEntity(
                                        text=triplet_data["head"]["text"],
                                        entity_type="UNKNOWN",
                                    ),
                                    relation=rel_type,
                                    tail=RERelationEntity(
                                        text=triplet_data["tail"]["text"],
                                        entity_type="UNKNOWN",
                                    ),
                                    score=score,
                                )
                            )

                    doc.results[self._task_id] = RERelationResult(triplets=triplets)

        return docs

    @override
    def consolidate(
        self, results: Sequence[gliner_.Result], docs_offsets: list[tuple[int, int]]
    ) -> Sequence[gliner_.Result]:
        consolidated_results: list[gliner_.Result] = []

        # Determine label scores for chunks per document.
        for doc_offset in docs_offsets:
            scores: dict[Any, float] = defaultdict(lambda: 0.0)
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
                                assert isinstance(entry, dict)
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

                            # Add entities from this chunk.
                            for entity in res[entity_type]:
                                # If single mode: we already have logic to track max_confidence below.
                                # But we also need to store all entities for general structure if not single?
                                # Actually, GLiNER structure mode output is a bit different.
                                relevant_entities_struct.append(entity)

                            if self._mode == "single":
                                for entity in res[entity_type]:
                                    # Calculate average confidence for the entity structure.
                                    assert isinstance(entity, dict)
                                    confidences = [
                                        field_val["confidence"]
                                        for field_val in entity.values()
                                        if "confidence" in field_val
                                    ]
                                    if confidences:
                                        avg_conf = sum(confidences) / len(confidences)
                                        if avg_conf > max_confidence:
                                            max_confidence = avg_conf
                                            # We need to preserve the confidence in the structure
                                            # so that integrate can pick it up.
                                            highest_confidence_entity = {entity_type: [entity]}

                    case gliner_.InferenceMode.relations:
                        # Collect all triplets from all chunks for this document.
                        # Simple unification for now.
                        relation_data = res.get(self._task_id) or res.get("relation_extraction", {})
                        assert isinstance(relation_data, dict)

                        for rel_type, triplets_list in relation_data.items():
                            for triplet in triplets_list:
                                key = (
                                    triplet["head"]["text"],
                                    rel_type,
                                    triplet["tail"]["text"],
                                    triplet["head"]["start"],
                                    triplet["tail"]["start"],
                                )
                                if key not in scores:  # Use scores dict as a seen set for convenience.
                                    scores[key] = 1.0
                                    all_entities_list.append({"relation": rel_type, **triplet})

            match self._inference_mode:
                case gliner_.InferenceMode.classification:
                    # Ensure that all labels have been assigned - GLiNER2 is sometimes negligent about this.
                    # We extract labels from our Pydantic signature fields (excluding 'score' if present).
                    labels = [name for name in self._pydantic_signature.model_fields if name != "score"]
                    for label in labels:
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

                case gliner_.InferenceMode.entities | gliner_.InferenceMode.structure | gliner_.InferenceMode.relations:
                    if self._inference_mode == gliner_.InferenceMode.structure and self._mode == "single":
                        if highest_confidence_entity:
                            consolidated_results.append(highest_confidence_entity)

                        else:
                            # Use the entity name from the pydantic signature (unwrapped).
                            model_cls = self._pydantic_signature

                            if "entity" in model_cls.model_fields:
                                model_cls = model_cls.model_fields["entity"].annotation
                                if typing.get_origin(model_cls) in (
                                    typing.Union,
                                    types.UnionType,
                                ):
                                    model_cls = [t for t in typing.get_args(model_cls) if t is not type(None)][0]

                            consolidated_results.append({model_cls.__name__: []})

                    elif all_entities_list:
                        if self._inference_mode == gliner_.InferenceMode.relations:
                            # Reconstruct the dict structure for integrate.
                            reconstructed: dict[str, list[Any]] = defaultdict(list)
                            for item in all_entities_list:
                                rel_type = item.pop("relation")
                                reconstructed[rel_type].append(item)
                            # Ensure we use the current task_id as the key so that integrate() can find it.
                            consolidated_results.append({self._task_id: dict(reconstructed)})
                        else:
                            consolidated_results.append(all_entities_list)
                    else:
                        consolidated_results.append(entities)

        return consolidated_results
