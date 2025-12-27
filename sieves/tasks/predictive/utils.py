"""Utilities for predictive tasks."""

from __future__ import annotations

import types
import typing
from collections.abc import Iterable, Sequence
from typing import Any, Literal, TypeVar, Union

import dspy
import gliner2.inference.engine
import pydantic

from sieves.model_wrappers import ModelType, dspy_, gliner_, huggingface_, outlines_

_EntityType = TypeVar("_EntityType", bound=pydantic.BaseModel)


def _get_literal_values(annotation: Any) -> list[str] | None:
    """Extract Literal values from an annotation, including those nested in Unions.

    :param annotation: The type annotation to inspect.
    :return: A list of Literal values if found, otherwise None.
    """
    origin = typing.get_origin(annotation)
    if origin is Literal:
        return list(typing.get_args(annotation))
    if origin in (Union, types.UnionType):
        for arg in typing.get_args(annotation):
            values = _get_literal_values(arg)
            if values:
                return values
    return None


def _extract_labels_from_model(model_cls: type[pydantic.BaseModel]) -> list[str]:
    """Extract candidate labels from a Pydantic model.

    Targets fields named 'label', 'entity_type', or 'relation' that use Literal types.
    Falls back to all non-excluded field names.

    :param model_cls: Pydantic model to extract labels from.
    :return: List of labels.
    """
    for field_name in ("label", "entity_type", "relation"):
        if field_name in model_cls.model_fields:
            values = _get_literal_values(model_cls.model_fields[field_name].annotation)
            if values:
                return values

    excluded = {"score", "reasoning"}
    return [name for name in model_cls.model_fields if name not in excluded]


def _pydantic_to_gliner(
    model_cls: type[pydantic.BaseModel], mode: str, **kwargs: Any
) -> gliner2.inference.engine.Schema | gliner2.inference.engine.StructureBuilder:
    """Convert Pydantic model to GliNER2 signature.

    :param model_cls: Pydantic model to convert.
    :param mode: GliNER2 mode (classification, entities, structure, relations).
    :param kwargs: Additional arguments for GliNER2 schema methods.
    :return: GliNER2 schema or structure builder.
    """
    schema = gliner2.inference.engine.Schema()

    if mode in ("classification", "entities", "relations"):
        labels = _extract_labels_from_model(model_cls)
        if mode == "classification":
            return schema.classification(labels=labels, **kwargs)
        if mode == "entities":
            return schema.entities(entity_types=labels, **kwargs)
        return schema.relations(relation_types=labels, **kwargs)

    if mode == "structure":

        def is_pydantic_model(t: Any) -> bool:
            return isinstance(t, type) and issubclass(t, pydantic.BaseModel)

        # Check for nested models.
        for field_name, field_info in model_cls.model_fields.items():
            annotation = field_info.annotation
            if is_pydantic_model(annotation) or any(is_pydantic_model(arg) for arg in typing.get_args(annotation)):
                raise ValueError(f"Nested Pydantic models are not supported for GliNER2. Field: {field_name}")

        struct = schema.structure(model_cls.__name__)
        for field_name, field_info in model_cls.model_fields.items():
            if field_name == "score":
                continue

            annotation = field_info.annotation
            choices = _get_literal_values(annotation)
            dtype = "str"

            if typing.get_origin(annotation) is list:
                dtype = "list"
                choices = _get_literal_values(typing.get_args(annotation)[0])

            struct.field(field_name, dtype=dtype, choices=choices) if choices else struct.field(field_name, dtype=dtype)

        return struct

    raise ValueError(f"Unsupported GliNER2 mode: {mode}")


def _convert_to_dspy(model_cls: type[pydantic.BaseModel]) -> type[dspy.Signature]:
    """Convert Pydantic model to DSPy Signature.

    :param model_cls: Pydantic model to convert.
    :return: DSPy Signature class.
    """
    memo: dict[type[pydantic.BaseModel], type[pydantic.BaseModel]] = {}

    def transform_type(ann: Any) -> Any:
        """Recursively transform types to be DSPy-compatible.

        This ensures that nested models and collections of models are converted to versions where
        field metadata is preserved using Annotated.

        :param ann: Type annotation to transform.
        :return: Transformed type annotation.
        """
        if isinstance(ann, type) and issubclass(ann, pydantic.BaseModel):
            return _model_to_dspy_compatible(ann)

        origin = typing.get_origin(ann)
        args = typing.get_args(ann)

        if origin in (list, Sequence, Iterable, typing.MutableSequence) and args:
            return list[transform_type(args[0])]  # type: ignore[invalid-type-form]
        if origin in (Union, types.UnionType) and args:
            return Union[tuple(transform_type(arg) for arg in args)]  # noqa: UP007

        return ann

    def _model_to_dspy_compatible(cls: type[pydantic.BaseModel]) -> type[pydantic.BaseModel]:
        """Convert a Pydantic model to one where all fields are Annotated with dspy.OutputField.

        This makes nested field descriptions visible to DSPy's prompt generation logic.

        :param cls: Pydantic model class to convert.
        :return: DSPy-compatible Pydantic model class.
        """
        if cls in memo:
            return memo[cls]

        fields_dict: dict[str, Any] = {}
        for name, field_info in cls.model_fields.items():
            new_ann = transform_type(field_info.annotation)
            # Wrap every nested field in Annotated with dspy.OutputField to ensure its description
            # is visible to DSPy's prompt generation logic.
            fields_dict[name] = (
                typing.Annotated[new_ann, dspy.OutputField(desc=field_info.description or "")],
                ... if field_info.is_required() else field_info.default,
            )

        dspy_compatible_model = pydantic.create_model(
            cls.__name__, **fields_dict, __base__=pydantic.BaseModel, __doc__=cls.__doc__
        )
        memo[cls] = dspy_compatible_model

        return dspy_compatible_model

    # Generate fields for the top-level DSPy Signature.
    fields = {"text": (str, dspy.InputField(desc="Input text to process."))}
    for name, field_info in model_cls.model_fields.items():
        # Transform the type recursively to include DSPy metadata at all levels.
        new_ann = transform_type(field_info.annotation)
        # Signature fields must be dspy.OutputField instances.
        fields[name] = (new_ann, dspy.OutputField(desc=field_info.description or ""))

    signature = type(
        model_cls.__name__,
        (dspy.Signature,),
        {
            "__annotations__": {k: v[0] for k, v in fields.items()},
            **{k: v[1] for k, v in fields.items()},
            "__doc__": model_cls.__doc__,
        },
    )
    assert issubclass(signature, dspy.Signature)

    return signature


def convert_to_signature(
    model_cls: type[pydantic.BaseModel],
    model_type: ModelType,
    **kwargs: Any,
) -> (
    dspy_.PromptSignature
    | type[dspy_.PromptSignature]
    | gliner_.PromptSignature
    | type[gliner_.PromptSignature]
    | huggingface_.PromptSignature
    | type[huggingface_.PromptSignature]
    | outlines_.PromptSignature
    | type[outlines_.PromptSignature]
):
    """Convert a Pydantic model to a framework-specific prompt signature.

    :param model_cls: Pydantic model to convert.
    :param model_type: Target model type/framework.
    :param kwargs: Additional framework-specific arguments.
    :return: Framework-specific prompt signature.
    :raises ValueError: If the model type is not supported.
    """
    match model_type:
        case ModelType.dspy:
            return _convert_to_dspy(model_cls)

        case ModelType.gliner:
            mode = kwargs.pop("mode", "structure")
            if mode == "classification" and "task" not in kwargs:
                kwargs["task"] = "classification"
            return _pydantic_to_gliner(model_cls, mode, **kwargs)

        case ModelType.huggingface:
            labels = _extract_labels_from_model(model_cls)
            is_classification = kwargs.get("mode") == "classification" or (
                labels and not all(label in model_cls.model_fields for label in labels)
            )
            return labels if is_classification else [name for name in model_cls.model_fields if name != "score"]

        case ModelType.outlines | ModelType.langchain:
            return _extract_labels_from_model(model_cls) if kwargs.get("inference_mode") == "choice" else model_cls

        case _:
            raise ValueError(f"Unsupported model type for signature conversion: {model_type}")
