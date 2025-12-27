# mypy: ignore-errors
from typing import Literal, Union, Any

import dspy
import gliner2.inference.engine
import pydantic
import pytest

from sieves.model_wrappers import ModelType
from sieves.tasks.predictive.utils import convert_to_signature

# --- Test Models ---

class SimpleExtraction(pydantic.BaseModel):
    """Simple extraction model."""
    name: str = pydantic.Field(description="The name of the person")
    age: int = pydantic.Field(description="The age of the person")

class ClassificationModel(pydantic.BaseModel):
    """Classification model."""
    label: Literal["A", "B", "C"] = pydantic.Field(description="The category")
    score: float

class MultiLabelModel(pydantic.BaseModel):
    """Multi-label model."""
    science: bool
    politics: bool
    sports: bool

# --- Helpers for comparison ---

def assert_dspy_sigs_equal(actual: type[dspy.Signature], expected: type[dspy.Signature]):
    """Deeply compare two DSPy signatures."""
    assert actual.input_fields.keys() == expected.input_fields.keys()
    assert actual.output_fields.keys() == expected.output_fields.keys()

    for k in actual.input_fields:
        assert actual.input_fields[k].json_schema_extra.get("desc") == expected.input_fields[k].json_schema_extra.get("desc")

    for k in actual.output_fields:
        assert actual.output_fields[k].json_schema_extra.get("desc") == expected.output_fields[k].json_schema_extra.get("desc")

    assert actual.__doc__ == expected.__doc__

def assert_gliner_schemas_equal(actual: Any, expected: Any):
    """Compare two GliNER schemas by their underlying dictionary representation."""
    if hasattr(actual, "_field_metadata"):
        assert actual._field_metadata == expected._field_metadata
    elif hasattr(actual, "schema"):
        if hasattr(actual.schema, "_field_metadata"):
            assert actual.schema._field_metadata == expected.schema._field_metadata
        else:
            assert actual.schema == expected.schema
    else:
        assert actual == expected

# --- Tests ---

def test_convert_to_dspy_full():
    # Define manually what we expect
    class ExpectedSig(dspy.Signature):
        """Simple extraction model."""
        text = dspy.InputField(desc="Input text to process.")
        name = dspy.OutputField(desc="The name of the person")
        age = dspy.OutputField(desc="The age of the person")

    actual = convert_to_signature(SimpleExtraction, ModelType.dspy)

    assert_dspy_sigs_equal(actual, ExpectedSig)

def test_convert_to_gliner_structure_full():
    # Expected manually constructed GliNER2 structure
    expected = gliner2.inference.engine.Schema().structure("SimpleExtraction")
    expected.field("name", dtype="str")
    expected.field("age", dtype="str")

    actual = convert_to_signature(SimpleExtraction, ModelType.gliner, mode="structure")

    assert_gliner_schemas_equal(actual, expected)

def test_convert_to_gliner_classification_full():
    # Expected manually constructed GliNER2 classification
    expected = gliner2.inference.engine.Schema().classification(
        task="classification",
        labels=["A", "B", "C"]
    )

    actual = convert_to_signature(ClassificationModel, ModelType.gliner, mode="classification")

    assert_gliner_schemas_equal(actual, expected)

def test_convert_to_huggingface_full():
    # Instruction: grab names of all fields that are not called 'score'.
    expected = ["A", "B", "C"]
    actual = convert_to_signature(ClassificationModel, ModelType.huggingface)
    assert actual == expected

    expected_multi = ["science", "politics", "sports"]
    actual_multi = convert_to_signature(MultiLabelModel, ModelType.huggingface)
    assert actual_multi == expected_multi

def test_convert_to_outlines_choice_full():
    # For choice mode, it should extract labels from Literal
    expected = ["A", "B", "C"]
    actual = convert_to_signature(ClassificationModel, ModelType.outlines, inference_mode="choice")
    assert actual == expected

def test_convert_to_gliner_entities_full():
    expected = gliner2.inference.engine.Schema().entities(
        entity_types=["science", "politics", "sports"]
    )

    actual = convert_to_signature(MultiLabelModel, ModelType.gliner, mode="entities")

    assert_gliner_schemas_equal(actual, expected)

def test_convert_to_gliner_structure_choices_full():
    class ChoiceModel(pydantic.BaseModel):
        category: Literal["X", "Y"]

    expected = gliner2.inference.engine.Schema().structure("ChoiceModel")
    expected.field("category", dtype="str", choices=["X", "Y"])

    actual = convert_to_signature(ChoiceModel, ModelType.gliner, mode="structure")

    assert_gliner_schemas_equal(actual, expected)
