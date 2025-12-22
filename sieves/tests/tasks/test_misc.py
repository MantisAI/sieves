# mypy: ignore-errors
import os
import pickle
import tempfile
import typing
from pathlib import Path

import chonkie
import datasets
import docling.document_converter
import dspy
import pydantic
import pytest
import transformers

from sieves import Doc, Pipeline, model_wrappers, tasks
from sieves.model_wrappers.utils import init_default_model
from sieves.tasks.utils import PydanticToHFDatasets


def test_custom_prompt_instructions() -> None:
    prompt_instructions = "This is a different prompt template."
    task = tasks.predictive.Classification(
        task_id="classifier",
        labels=["science", "politics"],
        model=transformers.pipeline(
            "zero-shot-classification", model="MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33"
        ),
        prompt_instructions=prompt_instructions,
    )
    assert task.prompt_template.strip().startswith(prompt_instructions)


def test_custom_prompt_signature_desc() -> None:
    prompt_instructions = "This is a different prompt signature description."
    task = tasks.predictive.Classification(
        task_id="classifier",
        labels=["science", "politics"],
        model=transformers.pipeline(
            "zero-shot-classification", model="MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33"
        ),
        prompt_instructions=prompt_instructions,
    )
    assert task.prompt_template.strip().startswith(prompt_instructions)


def test_pydantic_to_hf() -> None:
    """Test the conversion of various Pydantic objects to HF datasets.Features."""

    # Check non-nested properties.

    class Simple(pydantic.BaseModel):
        a: int
        b: str
        c: str | float
        d: tuple[int, float]
        e: str | None
        f: str | None  # noqa: UP007

    features = PydanticToHFDatasets.model_cls_to_features(Simple)
    assert all([key in features for key in ("a", "b")])
    assert features["a"].dtype == "int32"
    assert features["b"].dtype == "string"
    assert features["c"].dtype == "string"
    assert isinstance(features["d"], datasets.Sequence)
    assert features["d"].feature.dtype == "string"
    assert features["e"].dtype == "string"
    assert features["f"].dtype == "string"
    assert PydanticToHFDatasets.model_to_dict(None) is None
    dataset = datasets.Dataset.from_list(
        [PydanticToHFDatasets.model_to_dict(Simple(a=1, b="blub", c=0.3, d=(1, 0.4), e=None, f=None))],
        features=features,
    )
    assert list(dataset)[0] == {"a": 1, "b": "blub", "c": "0.3", "d": ["1", "0.4"], "e": None, "f": None}

    # With a list of primitives.

    class WithList(pydantic.BaseModel):
        a: int
        b: list[str]

    features = PydanticToHFDatasets.model_cls_to_features(WithList)
    assert all([key in features for key in ("a", "b")])
    assert features["a"].dtype == "int32"
    assert isinstance(features["b"], datasets.Sequence)
    assert features["b"].feature.dtype == "string"
    kwargs = {"a": 1, "b": ["blub", "blab"]}
    dataset = datasets.Dataset.from_list([PydanticToHFDatasets.model_to_dict(WithList(**kwargs))], features=features)
    assert list(dataset)[0] == kwargs

    # With a dictionary of primitives.

    class WithDict(pydantic.BaseModel):
        a: int
        b: dict[str, int]
        c: dict

    features = PydanticToHFDatasets.model_cls_to_features(WithDict)
    assert all([key in features for key in ("a", "b")])
    assert isinstance(features["b"], datasets.Sequence)
    assert isinstance(features["b"].feature, datasets.Features)
    assert isinstance(features["c"], datasets.Value)
    assert features["c"].dtype == "string"
    assert all([name in features["b"].feature for name in ("key", "value")])
    dataset = datasets.Dataset.from_list(
        [PydanticToHFDatasets.model_to_dict(WithDict(a=1, b={"blub": 2, "blab": 3}, c={"blib": 4}))],
        features=features
    )
    assert list(dataset)[0] == {'a': 1, 'b': {'key': ['blub', 'blab'], 'value': [2, 3]}, 'c': "{'blib': 4}"}

    # With nested Pydantic models.

    class SubModel(pydantic.BaseModel):
        c: int

    class WithPydModel(pydantic.BaseModel):
        a: bool
        b: SubModel

    features = PydanticToHFDatasets.model_cls_to_features(WithPydModel)
    assert all([key in features for key in ("a", "b")])
    assert isinstance(features["a"], datasets.Value)
    assert features["a"].dtype == "bool"
    assert isinstance(features["b"], dict)
    assert "c" in features["b"]
    assert isinstance(features["b"]["c"], datasets.Value)
    assert features["b"]["c"].dtype == "int32"
    dataset = datasets.Dataset.from_list(
        [PydanticToHFDatasets.model_to_dict(WithPydModel(a=True, b=SubModel(c=3)))], features=features
    )
    assert list(dataset)[0] == {"a": True, "b": {"c": 3}}

    # With a dictionary of nested Pydantic models.

    class NestedModel(pydantic.BaseModel):
        sub_models: list[SubModel]

    class WithNestedDictPydModel(pydantic.BaseModel):
        a: bool
        b: dict[str, NestedModel]

    features = PydanticToHFDatasets.model_cls_to_features(WithNestedDictPydModel)
    assert all([key in features for key in ("a", "b")])
    assert isinstance(features["a"], datasets.Value)
    assert features["a"].dtype == "bool"
    assert isinstance(features["b"], datasets.Sequence)
    assert isinstance(features["b"].feature, datasets.Features)
    assert isinstance(features["b"].feature["key"], datasets.Value)
    assert features["b"].feature["key"].dtype == "string"
    assert isinstance(features["b"].feature["value"], datasets.Features)
    assert isinstance(features["b"].feature["value"]["sub_models"], datasets.Sequence)
    assert isinstance(features["b"].feature["value"]["sub_models"].feature, datasets.Features)
    assert isinstance(features["b"].feature["value"]["sub_models"].feature["c"], datasets.Value)
    assert features["b"].feature["value"]["sub_models"].feature["c"].dtype == "int32"


def test_pydantic_to_hf_nested_and_optional() -> None:
    """Test nested and optional models in PydanticToHFDatasets."""

    # Nested models in list
    class SubModel(pydantic.BaseModel):
        x: int

    class MainModelList(pydantic.BaseModel):
        subs: list[SubModel]

    model_list = MainModelList(subs=[SubModel(x=1), SubModel(x=2)])
    features_list = PydanticToHFDatasets.model_cls_to_features(MainModelList)
    result_list = PydanticToHFDatasets.model_to_dict(model_list)
    assert result_list == {"subs": [{"x": 1}, {"x": 2}]}
    ds_list = datasets.Dataset.from_list([result_list], features=features_list)
    assert ds_list[0] == {"subs": {"x": [1, 2]}}

    # Nested models in dict
    class MainModelDict(pydantic.BaseModel):
        subs: dict[str, SubModel]

    model_dict = MainModelDict(subs={"a": SubModel(x=1), "b": SubModel(x=2)})
    features_dict = PydanticToHFDatasets.model_cls_to_features(MainModelDict)
    result_dict = PydanticToHFDatasets.model_to_dict(model_dict)
    assert result_dict == {"subs": [{"key": "a", "value": {"x": 1}}, {"key": "b", "value": {"x": 2}}]}
    ds_dict = datasets.Dataset.from_list([result_dict], features=features_dict)
    assert ds_dict[0] == {"subs": {"key": ["a", "b"], "value": [{"x": 1}, {"x": 2}]}}

    # Optional nested model
    class MainModelOptional(pydantic.BaseModel):
        sub: SubModel | None

    model_opt = MainModelOptional(sub=SubModel(x=1))
    features_opt = PydanticToHFDatasets.model_cls_to_features(MainModelOptional)
    assert isinstance(features_opt["sub"], datasets.Features)
    result_opt = PydanticToHFDatasets.model_to_dict(model_opt)
    assert result_opt == {"sub": {"x": 1}}
    ds_opt = datasets.Dataset.from_list([result_opt], features=features_opt)
    assert ds_opt[0] == {"sub": {"x": 1}}

    # Deeply nested objects
    class Level3(pydantic.BaseModel):
        val: int

    class Level2(pydantic.BaseModel):
        l3: Level3

    class Level1(pydantic.BaseModel):
        l2: list[Level2]

    model_deep = Level1(l2=[Level2(l3=Level3(val=42))])
    features_deep = PydanticToHFDatasets.model_cls_to_features(Level1)
    result_deep = PydanticToHFDatasets.model_to_dict(model_deep)
    assert result_deep == {"l2": [{"l3": {"val": 42}}]}
    ds_deep = datasets.Dataset.from_list([result_deep], features=features_deep)
    assert ds_deep[0] == {"l2": {"l3": [{"val": 42}]}}
