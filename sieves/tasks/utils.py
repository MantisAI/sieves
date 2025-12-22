"""Utilities."""

from __future__ import annotations

import abc
import types
import typing
from typing import Any

import datasets
import pydantic


class PydanticToHFDatasets(abc.ABC):
    """Collection of utilities for converting Pydantic models (types and instances) to HF's `datasets.Dataset`."""

    _PRIMITIVES_MAP: dict[type, str] = {
        str: "string",
        int: "int32",
        float: "float32",
        bool: "bool",
    }

    @classmethod
    def model_cls_to_features(cls, entity_type: type[pydantic.BaseModel]) -> datasets.Features:
        """Given a Pydantic model, build a `datasets.Sequence` of features that match its fields.

        :param entity_type: The Pydantic model class to convert.
        :return: A `datasets.Features` instance for use in a Hugging Face `datasets.Dataset`.
        """
        field_features: dict[str, datasets.Value | datasets.Sequence | datasets.Features] = {}

        for field_name, field_info in entity_type.model_fields.items():
            field_features[field_name] = cls._annotation_to_values(field_info.annotation)

        return datasets.Features(field_features)

    @classmethod
    def _annotation_to_values(cls, annotation: Any) -> datasets.Value | datasets.Sequence | datasets.Features:
        """Convert a type annotation to a Hugging Face `datasets` feature.

        :param annotation: The type annotation to convert (e.g., str, list[int], or a Pydantic model).
        :return: A Hugging Face dataset feature instance generated from the specified annotation.
        """
        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)

        # 1) Nested Pydantic Model.
        if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
            return cls.model_cls_to_features(annotation)

        # 2) Sequences (list, tuple).
        if origin in (list, tuple):
            return cls._handle_sequence_annotation(args)

        # 3) Dictionaries.
        if origin is dict:
            return cls._handle_dict_annotation(args)

        # 4) Union / Optional.
        if origin in (typing.Union, getattr(types, "UnionType", None)):
            return cls._handle_union_annotation(args)

        # 5) Primitives & Fallback.
        return datasets.Value(cls._PRIMITIVES_MAP.get(annotation, "string"))

    @classmethod
    def _handle_sequence_annotation(cls, args: tuple[Any, ...]) -> datasets.Sequence:
        """Handle list[...] and tuple[...] annotations.

        :param args: The type arguments of the sequence annotation.
        :return: A `datasets.Sequence` feature.
        """
        if len(args) == 1:
            return datasets.Sequence(cls._annotation_to_values(args[0]))
        # Fallback for heterogeneous tuples or untyped lists.
        return datasets.Sequence(datasets.Value("string"))

    @classmethod
    def _handle_dict_annotation(cls, args: tuple[Any, ...]) -> datasets.Sequence | datasets.Value:
        """Handle dict[...] annotations.

        :param args: The type arguments of the dictionary annotation.
        :return: A `datasets.Sequence` for typed string-key dicts, or a `datasets.Value` fallback.
        """
        if len(args) == 2 and args[0] is str:
            # For dict[str, T], store as a sequence of key-value pairs.
            return datasets.Sequence(
                feature=datasets.Features(
                    {"key": datasets.Value("string"), "value": cls._annotation_to_values(args[1])}
                )
            )
        # Fallback for non-string keys or untyped dicts.
        return datasets.Value("string")

    @classmethod
    def _handle_union_annotation(cls, args: tuple[Any, ...]) -> datasets.Value | datasets.Sequence | datasets.Features:
        """Handle Union and Optional annotations.

        :param args: The type arguments of the Union annotation.
        :return: The feature for the underlying type if Optional, otherwise a string fallback.
        """
        underlying_type = cls._get_underlying_optional_type(args)
        if underlying_type:
            return cls._annotation_to_values(underlying_type)
        return datasets.Value("string")

    @classmethod
    def model_to_dict(cls, model: pydantic.BaseModel | None) -> dict[str, Any] | None:
        """Convert a Pydantic model instance to a dict aligned with the HF dataset schema.

        :param model: The Pydantic model instance to convert.
        :return: A dictionary representation of the model instance, or None if the input is None.
        """
        if model is None:
            return None

        if isinstance(model, pydantic.BaseModel):
            out: dict[str, Any] = {}
            for field_name, field_info in type(model).model_fields.items():
                value = getattr(model, field_name)
                out[field_name] = cls._convert_value_for_dataset(value, field_info.annotation)
            return out

        return model  # type: ignore[return-value]

    @classmethod
    def _convert_value_for_dataset(cls, value: Any, annotation: Any) -> Any:
        """Recursively convert a value to something that fits the HF dataset row format.

        :param value: The value to convert.
        :param annotation: The type annotation associated with the value.
        :return: The converted value compatible with Hugging Face datasets.
        """
        if value is None:
            return None

        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)

        # 1) Nested Pydantic Model.
        if isinstance(value, pydantic.BaseModel):
            return cls.model_to_dict(value)

        # 2) Sequences (list, tuple).
        if origin in (list, tuple):
            return cls._handle_sequence_value(value, args)

        # 3) Dictionaries.
        if origin is dict:
            return cls._handle_dict_value(value, args)

        # 4) Union / Optional.
        if origin in (typing.Union, getattr(types, "UnionType", None)):
            return cls._handle_union_value(value, args)

        # 5) Primitives & fallback.
        if annotation in (str, int, float, bool):
            return value
        return str(value)

    @classmethod
    def _handle_sequence_value(cls, value: Any, args: tuple[Any, ...]) -> list[Any] | str:
        """Handle sequence values.

        :param value: The sequence value to convert.
        :param args: The type arguments of the sequence annotation.
        :return: A list of converted values, or a string representation fallback.
        """
        if not isinstance(value, (list, tuple)):  # noqa: UP038
            return str(value)

        if len(args) == 1:
            return [cls._convert_value_for_dataset(v, args[0]) for v in value]

        return [str(v) for v in value]

    @classmethod
    def _handle_dict_value(cls, value: Any, args: tuple[Any, ...]) -> list[dict[str, Any]] | str:
        """Handle dictionary values.

        :param value: The dictionary value to convert.
        :param args: The type arguments of the dictionary annotation.
        :return: A list of key-value pair dictionaries, or a string representation fallback.
        """
        if not isinstance(value, dict):
            return str(value)

        if len(args) == 2 and args[0] is str:
            return [{"key": str(k), "value": cls._convert_value_for_dataset(v, args[1])} for k, v in value.items()]

        return str(value)

    @classmethod
    def _handle_union_value(cls, value: Any, args: tuple[Any, ...]) -> Any:
        """Handle Union and Optional values.

        :param value: The value to convert.
        :param args: The type arguments of the Union annotation.
        :return: The converted value if Optional, otherwise a string representation fallback.
        """
        underlying_type = cls._get_underlying_optional_type(args)
        if underlying_type:
            return cls._convert_value_for_dataset(value, underlying_type)
        return str(value)

    @classmethod
    def _get_underlying_optional_type(cls, args: tuple[Any, ...]) -> Any | None:
        """Extract T from Optional[T] / Union[T, None].

        :param args: The type arguments of the Union annotation.
        :return: The underlying type T if it is an Optional[T], otherwise None.
        """
        non_none_args = [arg for arg in args if arg is not type(None)]
        return non_none_args[0] if len(non_none_args) == 1 else None
