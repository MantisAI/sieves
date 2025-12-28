# mypy: ignore-errors
from typing import Literal, Union, Any

import dspy
import gliner2.inference.engine
import pydantic
import pytest

from sieves.model_wrappers import ModelType
from sieves.tasks.predictive.utils import convert_to_signature

# --- Helper for comparison (reused from test_signature_utils.py) ---

def assert_dspy_sigs_equal(actual: type[dspy.Signature], expected: type[dspy.Signature]):
    assert actual.input_fields.keys() == expected.input_fields.keys()
    assert actual.output_fields.keys() == expected.output_fields.keys()

    for k in actual.input_fields:
        assert actual.input_fields[k].json_schema_extra.get("desc") == expected.input_fields[k].json_schema_extra.get("desc")

    for k in actual.output_fields:
        actual_desc = actual.output_fields[k].json_schema_extra.get("desc")
        expected_desc = expected.output_fields[k].json_schema_extra.get("desc")
        assert actual_desc == expected_desc

    assert actual.__doc__ == expected.__doc__

def assert_gliner_schemas_equal(actual: Any, expected: Any):
    if hasattr(actual, "_field_metadata"):
        assert actual._field_metadata == expected._field_metadata
    elif hasattr(actual, "schema"):
        if hasattr(actual.schema, "_field_metadata"):
            assert actual.schema._field_metadata == expected.schema._field_metadata
        else:
            assert actual.schema == expected.schema
    else:
        assert actual == expected

# --- Task-specific Tests ---

def test_classification_migration():
    # 1. Define Unified Pydantic Prompt Signature
    class ClassificationOutput(pydantic.BaseModel):
        """Classify the text into one of the provided labels."""
        label: Literal["science", "politics", "sports"] = pydantic.Field(description="The predicted label")
        score: float = pydantic.Field(description="Confidence score")

    # 2. Define Expected Output Signatures

    # DSPy
    class ExpectedDSPy(dspy.Signature):
        """Classify the text into one of the provided labels."""
        text = dspy.InputField(desc="Input text to process.")
        label = dspy.OutputField(desc="The predicted label")
        score = dspy.OutputField(desc="Confidence score")

    # GliNER (classification mode)
    expected_gliner = gliner2.inference.engine.Schema().classification(
        task="classification",
        labels=["science", "politics", "sports"]
    )

    # HuggingFace (labels)
    expected_hf = ["science", "politics", "sports"]

    # Outlines (choice mode)
    expected_outlines_choice = ["science", "politics", "sports"]

    # 3. Verify
    assert_dspy_sigs_equal(convert_to_signature(ClassificationOutput, ModelType.dspy), ExpectedDSPy)
    assert_gliner_schemas_equal(convert_to_signature(ClassificationOutput, ModelType.gliner, mode="classification"), expected_gliner)
    assert convert_to_signature(ClassificationOutput, ModelType.huggingface) == expected_hf
    assert convert_to_signature(ClassificationOutput, ModelType.outlines, inference_mode="choice") == expected_outlines_choice


def test_classification_multi_label_migration():
    # 1. Define Unified Pydantic Prompt Signature
    class MultiLabelClassificationOutput(pydantic.BaseModel):
        """Perform multi-label classification."""
        science: float = pydantic.Field(description="Score for science category")
        politics: float = pydantic.Field(description="Score for politics category")
        sports: float = pydantic.Field(description="Score for sports category")

    # 2. Define Expected Output Signatures

    # DSPy (Unified schema leads to flat output fields)
    class ExpectedDSPy(dspy.Signature):
        """Perform multi-label classification."""
        text = dspy.InputField(desc="Input text to process.")
        science = dspy.OutputField(desc="Score for science category")
        politics = dspy.OutputField(desc="Score for politics category")
        sports = dspy.OutputField(desc="Score for sports category")

    # GliNER (classification mode)
    expected_gliner = gliner2.inference.engine.Schema().classification(
        task="classification",
        labels=["science", "politics", "sports"]
    )

    # HuggingFace (labels)
    expected_hf = ["science", "politics", "sports"]

    # Outlines (JSON mode)
    expected_outlines = MultiLabelClassificationOutput

    # 3. Verify
    assert_dspy_sigs_equal(convert_to_signature(MultiLabelClassificationOutput, ModelType.dspy), ExpectedDSPy)
    assert_gliner_schemas_equal(convert_to_signature(MultiLabelClassificationOutput, ModelType.gliner, mode="classification"), expected_gliner)
    assert convert_to_signature(MultiLabelClassificationOutput, ModelType.huggingface) == expected_hf
    assert convert_to_signature(MultiLabelClassificationOutput, ModelType.outlines) == expected_outlines


def test_information_extraction_migration():
    # 1. Define Unified Pydantic Prompt Signature (for one entity)
    class Person(pydantic.BaseModel):
        """Extract person details."""
        name: str = pydantic.Field(description="Full name")
        age: int = pydantic.Field(description="Age in years")

    # 2. Define Expected Output Signatures

    # DSPy
    class ExpectedDSPy(dspy.Signature):
        """Extract person details."""
        text = dspy.InputField(desc="Input text to process.")
        name = dspy.OutputField(desc="Full name")
        age = dspy.OutputField(desc="Age in years")

    # GliNER (structure mode)
    expected_gliner = gliner2.inference.engine.Schema().structure("Person")
    expected_gliner.field("name", dtype="str")
    expected_gliner.field("age", dtype="str")

    # 3. Verify
    assert_dspy_sigs_equal(convert_to_signature(Person, ModelType.dspy), ExpectedDSPy)
    assert_gliner_schemas_equal(convert_to_signature(Person, ModelType.gliner, mode="structure"), expected_gliner)


def test_ner_migration():
    # 1. Define Unified Pydantic Prompt Signature
    class NEROutput(pydantic.BaseModel):
        """Identify named entities in the text."""
        label: Literal["PER", "LOC", "ORG"] = pydantic.Field(description="Entity type")

    # 2. Define Expected Output Signatures

    # GliNER (entities mode)
    expected_gliner = gliner2.inference.engine.Schema().entities(
        entity_types=["PER", "LOC", "ORG"]
    )

    # 3. Verify
    assert_gliner_schemas_equal(convert_to_signature(NEROutput, ModelType.gliner, mode="entities"), expected_gliner)


def test_pii_masking_migration():
    # 1. Define Unified Pydantic Prompt Signature
    class PIIOutput(pydantic.BaseModel):
        """Mask PII entities in the text and list them."""
        masked_text: str = pydantic.Field(description="Text with PII replaced by placeholders")
        pii_entities: list[str] = pydantic.Field(description="List of detected PII entities")

    # 2. Define Expected Output Signatures

    # DSPy
    class ExpectedDSPy(dspy.Signature):
        """Mask PII entities in the text and list them."""
        text = dspy.InputField(desc="Input text to process.")
        masked_text = dspy.OutputField(desc="Text with PII replaced by placeholders")
        pii_entities = dspy.OutputField(desc="List of detected PII entities")

    # 3. Verify
    assert_dspy_sigs_equal(convert_to_signature(PIIOutput, ModelType.dspy), ExpectedDSPy)


def test_question_answering_migration():
    # 1. Define Unified Pydantic Prompt Signature
    class QAOutput(pydantic.BaseModel):
        """Answer the question based on the text."""
        answer: str = pydantic.Field(description="The answer to the question")
        score: float = pydantic.Field(description="Confidence score")

    # 2. Define Expected Output Signatures

    # DSPy
    class ExpectedDSPy(dspy.Signature):
        """Answer the question based on the text."""
        text = dspy.InputField(desc="Input text to process.")
        answer = dspy.OutputField(desc="The answer to the question")
        score = dspy.OutputField(desc="Confidence score")

    # 3. Verify
    assert_dspy_sigs_equal(convert_to_signature(QAOutput, ModelType.dspy), ExpectedDSPy)


def test_relation_extraction_migration():
    # 1. Define Unified Pydantic Prompt Signature (for one triplet)
    class RelationOutput(pydantic.BaseModel):
        """Extract relations between entities."""
        head: str = pydantic.Field(description="Subject entity")
        relation: Literal["WORKS_AT", "LIVES_IN"] = pydantic.Field(description="Relation type")
        tail: str = pydantic.Field(description="Object entity")

    # 2. Define Expected Output Signatures

    # GliNER (relations mode)
    expected_gliner = gliner2.inference.engine.Schema().relations(
        relation_types=["head", "relation", "tail"] # Per instruction: all but score
    )

    # 3. Verify
    assert_gliner_schemas_equal(convert_to_signature(RelationOutput, ModelType.gliner, mode="relations"), expected_gliner)


def test_sentiment_analysis_migration():
    # 1. Define Unified Pydantic Prompt Signature
    class SentimentOutput(pydantic.BaseModel):
        """Analyze sentiment per aspect."""
        sentiment_per_aspect: dict[str, float] = pydantic.Field(description="Score per aspect")
        score: float = pydantic.Field(description="Overall confidence")

    # 2. Define Expected Output Signatures

    # DSPy
    class ExpectedDSPy(dspy.Signature):
        """Analyze sentiment per aspect."""
        text = dspy.InputField(desc="Input text to process.")
        sentiment_per_aspect = dspy.OutputField(desc="Score per aspect")
        score = dspy.OutputField(desc="Overall confidence")

    # 3. Verify
    assert_dspy_sigs_equal(convert_to_signature(SentimentOutput, ModelType.dspy), ExpectedDSPy)


def test_summarization_migration():
    # 1. Define Unified Pydantic Prompt Signature
    class SummarizationOutput(pydantic.BaseModel):
        """Summarize the text."""
        summary: str = pydantic.Field(description="The generated summary")
        score: float = pydantic.Field(description="Confidence score")

    # 2. Define Expected Output Signatures

    # DSPy
    class ExpectedDSPy(dspy.Signature):
        """Summarize the text."""
        text = dspy.InputField(desc="Input text to process.")
        summary = dspy.OutputField(desc="The generated summary")
        score = dspy.OutputField(desc="Confidence score")

    # 3. Verify
    assert_dspy_sigs_equal(convert_to_signature(SummarizationOutput, ModelType.dspy), ExpectedDSPy)


def test_translation_migration():
    # 1. Define Unified Pydantic Prompt Signature
    class TranslationOutput(pydantic.BaseModel):
        """Translate the text."""
        translation: str = pydantic.Field(description="The translated text")
        score: float = pydantic.Field(description="Confidence score")

    # 2. Define Expected Output Signatures

    # DSPy
    class ExpectedDSPy(dspy.Signature):
        """Translate the text."""
        text = dspy.InputField(desc="Input text to process.")
        translation = dspy.OutputField(desc="The translated text")
        score = dspy.OutputField(desc="Confidence score")

    # 3. Verify
    assert_dspy_sigs_equal(convert_to_signature(TranslationOutput, ModelType.dspy), ExpectedDSPy)
