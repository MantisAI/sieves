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

from sieves import Doc, Engine, Pipeline, engines, tasks
from sieves.tasks.utils import PydanticToHFDatasets


def test_custom_prompt_template() -> None:
    prompt_template = "This is a different prompt template."
    engine = Engine(model=dspy.LM("claude-3-haiku-20240307", api_key=os.environ["ANTHROPIC_API_KEY"]))
    task = tasks.predictive.Classification(
        task_id="classifier",
        labels=["science", "politics"],
        engine=engine,
        prompt_template=prompt_template,
    )
    assert task.prompt_template == prompt_template


def test_custom_prompt_signature_desc() -> None:
    prompt_sig_desc = "This is a different prompt signature description."
    engine = Engine(model=dspy.LM("claude-3-haiku-20240307", api_key=os.environ["ANTHROPIC_API_KEY"]))
    task = tasks.predictive.Classification(
        task_id="classifier",
        labels=["science", "politics"],
        engine=engine,
        prompt_signature_desc=prompt_sig_desc,
    )
    assert task.prompt_signature_description == prompt_sig_desc


def test_run_readme_example_short() -> None:
    # Define documents by text or URI.
    docs = [Doc(text="Special relativity applies to all physical phenomena in the absence of gravity.")]

    # Create pipeline with tasks.
    pipe = Pipeline(
        # Run classification on provided document.
        tasks.predictive.Classification(labels=["science", "politics"], engine=Engine())
    )

    # Run pipe and output results.
    docs = list(pipe(docs))
    print(docs[0].results["Classification"])


@pytest.mark.slow
@pytest.mark.parametrize(
    "batch_engine",
    [engines.EngineType.glix],
    indirect=True,
)
def test_run_readme_example_long(batch_engine, tokenizer) -> None:
    # Define documents by text or URI.
    # Readme example downlodads https://arxiv.org/pdf/2408.09869, but we'll use a local PDF here to speed up the test.
    docs = [Doc(uri=Path(__file__).parent.parent / "assets" / "1204.0162v2.pdf")]

    # Create pipeline with tasks.
    pipe = Pipeline(
        [
            # Add document parsing task.
            tasks.Ingestion(export_format="markdown"),
            # Add chunking task to ensure we don't exceed our model's context window.
            tasks.Chunking(chonkie.TokenChunker(tokenizer)),
            # Run classification on provided document.
            tasks.predictive.Classification(labels=["science", "politics"], engine=batch_engine),
        ]
    )

    # Run pipe and output results.
    docs = list(pipe(docs))

    # Serialize pipeline and docs.
    with tempfile.NamedTemporaryFile(suffix=".yml") as tmp_pipeline_file:
        with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp_docs_file:
            pipe.dump(tmp_pipeline_file.name)
            with open(tmp_docs_file.name, "wb") as f:
                pickle.dump(docs, f)

            # To load a pipeline and docs from disk:
            loaded_pipe = Pipeline.load(
                tmp_pipeline_file.name,
                (
                    {"converter": docling.document_converter.DocumentConverter()},
                    {"chunker": chonkie.TokenChunker(tokenizer)},
                    {"engine": {"model": batch_engine.model}},
                ),
            )

            pipe_config = pipe.serialize().model_dump()
            assert pipe_config["tasks"]["value"][2]["fewshot_examples"]["value"] == ()
            pipe_config["tasks"]["value"][2]["fewshot_examples"]["value"] = []
            assert loaded_pipe.serialize().model_dump() == pipe_config

            with open(tmp_docs_file.name, "rb") as f:
                loaded_docs = pickle.load(f)
            assert len(loaded_docs) == len(docs)
            assert all([(ld == d for ld, d in zip(loaded_docs, docs))])


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
        [PydanticToHFDatasets.model_to_dict(WithDict(a=1, b={"blub": 2, "blab": 3}, c={"blib": 4}))], features=features
    )
    assert list(dataset)[0] == {"a": 1, "b": {"key": ["blub", "blab"], "value": [2, 3]}, "c": "{'blib': 4}"}

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
