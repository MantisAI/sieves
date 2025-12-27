"""Utilities for predictive tasks."""

from __future__ import annotations

import types
import typing
from typing import Any, Literal, TypeVar, Union

import dspy
import gliner2.inference.engine
import pydantic

from sieves.model_wrappers import ModelType, dspy_, gliner_, huggingface_, outlines_

_EntityType = TypeVar("_EntityType", bound=pydantic.BaseModel)


def _extract_labels_from_model(model_cls: type[pydantic.BaseModel]) -> list[str]:
    """Extract candidate labels from a Pydantic model.

    :param model_cls: Pydantic model to extract labels from.
    :return: List of labels.
    """
    # 1. Check for a 'label' field with Literal.
    if "label" in model_cls.model_fields:
        field = model_cls.model_fields["label"]
        annotation = field.annotation

        # Unpack Optional/Union
        if typing.get_origin(annotation) in (Union, types.UnionType):
            args = typing.get_args(annotation)
            for arg in args:
                if typing.get_origin(arg) is Literal:
                    return list(typing.get_args(arg))

        if typing.get_origin(annotation) is Literal:
            return list(typing.get_args(annotation))

    # 2. Fallback: all field names that aren't excluded.
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

    if mode == "classification":
        labels = _extract_labels_from_model(model_cls)
        return schema.classification(labels=labels, **kwargs)

    elif mode == "entities":
        labels = _extract_labels_from_model(model_cls)
        return schema.entities(entity_types=labels, **kwargs)

    elif mode == "structure":

        def is_pydantic_model(t: Any) -> bool:
            return isinstance(t, type) and issubclass(t, pydantic.BaseModel)

        # Check for nested models.
        for field_name, field_info in model_cls.model_fields.items():
            annotation = field_info.annotation
            if is_pydantic_model(annotation):
                raise ValueError(f"Nested Pydantic models are not supported for GliNER2. Field: {field_name}")

            args = typing.get_args(annotation)
            if any(is_pydantic_model(arg) for arg in args):
                raise ValueError(f"Nested Pydantic models are not supported for GliNER2. Field: {field_name}")

        struct = schema.structure(model_cls.__name__)
        for field_name, field_info in model_cls.model_fields.items():
            if field_name == "score":
                continue

            annotation = field_info.annotation
            origin = typing.get_origin(annotation)

            dtype = "str"
            choices = None

            if origin is Literal:
                choices = list(typing.get_args(annotation))
            elif origin is list:
                dtype = "list"
                inner_args = typing.get_args(annotation)
                if inner_args and typing.get_origin(inner_args[0]) is Literal:
                    choices = list(typing.get_args(inner_args[0]))

            if choices:
                struct.field(field_name, dtype=dtype, choices=choices)
            else:
                struct.field(field_name, dtype=dtype)

        return struct

    elif mode == "relations":
        labels = _extract_labels_from_model(model_cls)
        return schema.relations(relation_types=labels, **kwargs)

    else:
        raise ValueError(f"Unsupported GliNER2 mode: {mode}")


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
            # Manually create DSPy Signature.
            fields = {
                "text": (str, dspy.InputField(desc="Input text to process.")),
            }
            for name, field_info in model_cls.model_fields.items():
                # Map description to desc.
                desc = field_info.description or ""
                fields[name] = (field_info.annotation, dspy.OutputField(desc=desc))

            # Create Signature subclass.
            sig = type(
                model_cls.__name__,
                (dspy.Signature,),
                {
                    "__annotations__": {k: v[0] for k, v in fields.items()},
                    **{k: v[1] for k, v in fields.items()},
                    "__doc__": model_cls.__doc__,
                },
            )

            return sig

        case ModelType.gliner:
            mode = kwargs.get("mode", "structure")
            # For classification, GliNER needs a 'task' name.
            if mode == "classification" and "task" not in kwargs:
                kwargs["task"] = "classification"

            return _pydantic_to_gliner(model_cls, mode, **{k: v for k, v in kwargs.items() if k != "mode"})

        case ModelType.huggingface:
            # Per instructions: grab names of all fields that are not called 'score'.
            return [name for name in model_cls.model_fields if name != "score"]

        case ModelType.outlines | ModelType.langchain:
            # Check if we should extract labels for choice mode.
            if kwargs.get("inference_mode") == "choice":
                return _extract_labels_from_model(model_cls)

            return model_cls

        case _:
            raise ValueError(f"Unsupported model type for signature conversion: {model_type}")
