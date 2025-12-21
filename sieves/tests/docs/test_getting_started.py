"""
Test file containing examples for the Getting Started guide.

These code blocks are referenced in docs/guides/getting_started.md using snippet injection.

Usage in markdown:
    ```python
    --8<-- "sieves/tests/docs/test_getting_started.py:basic-classification"
    ```
"""
import os

import chonkie
import dspy
import pytest
import outlines
import pydantic
from pathlib import Path

import tokenizers

from sieves import Doc, Pipeline, tasks
from sieves.model_wrappers import ModelType


def test_label_descriptions_example(small_outlines_model):
    """Test classification with label descriptions."""
    model = small_outlines_model  # For testing, use fixture.

    # --8<-- [start:label-descriptions]
    import outlines
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sieves import Pipeline, tasks, Doc

    doc = Doc(text="Special relativity applies to all physical phenomena in the absence of gravity.")
    model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    model = outlines.models.from_transformers(
        AutoModelForCausalLM.from_pretrained(model_name),
        AutoTokenizer.from_pretrained(model_name)
    )

    # Use dict format to provide descriptions.
    pipeline = Pipeline([
        tasks.predictive.Classification(
            labels={
                "science": "Scientific topics including physics, biology, chemistry, and natural sciences",
                "politics": "Political news, government affairs, elections, and policy discussions"
            },
            model=model,
        )
    ])

    for doc in pipeline([doc]):
        print(doc.results)
    # --8<-- [end:label-descriptions]

    assert doc.results is not None


def test_doc_creation_examples():
    """Test various ways to create documents."""
    # --8<-- [start:doc-from-text]
    from sieves import Doc

    # From text.
    doc = Doc(text="Your text here")
    # --8<-- [end:doc-from-text]
    assert doc.text == "Your text here"

    # --8<-- [start:doc-with-metadata]
    # With metadata.
    doc = Doc(
        text="Your text here",
        meta={"source": "example", "date": "2025-01-31"}
    )
    # --8<-- [end:doc-with-metadata]
    assert doc.meta["source"] == "example"


def test_doc_from_uri_example():
    """Test creating document from URI (requires ingestion extra)."""
    # --8<-- [start:doc-from-uri]
    from sieves import Doc

    # From a file (requires docling).
    doc = Doc(uri="path/to/your/file.pdf")
    # --8<-- [end:doc-from-uri]
    # Note: This would fail without a real file, hence the skip marker.


def test_advanced_pipeline_example(example_chunker, small_outlines_model):
    """Test advanced pipeline with chunking and extraction."""
    # --8<-- [start:advanced-pipeline]
    import chonkie
    import outlines
    import tokenizers
    import pydantic
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sieves import Doc, Pipeline, tasks

    # Create a tokenizer for chunking.
    tokenizer = tokenizers.Tokenizer.from_pretrained("bert-base-uncased")

    # Initialize components.
    chunker = tasks.Chunking(
        chunker=chonkie.TokenChunker(tokenizer, chunk_size=512, chunk_overlap=50)
    )

    # Choose a model for information extraction.
    model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    model = outlines.models.from_transformers(
        AutoModelForCausalLM.from_pretrained(model_name),
        AutoTokenizer.from_pretrained(model_name)
    )

    # Define the structure of information you want to extract.
    class PersonInfo(pydantic.BaseModel, frozen=True):
        name: str
        age: int | None = None
        occupation: str | None = None

    # Create an information extraction task.
    extractor = tasks.predictive.InformationExtraction(
        entity_type=PersonInfo,
        model=model,
    )

    # Create the pipeline (use + for succinct chaining).
    pipeline = chunker + extractor

    # Process a document.
    doc = Doc(text="Marie Curie died at the age of 66 years.")
    results = list(pipeline([doc]))

    # Access the extracted information.
    for result in results:
        print(result.results["InformationExtraction"])
    # --8<-- [end:advanced-pipeline]

    assert len(results) > 0


def test_model_settings_example(small_transformer_model):
    """Test ModelSettings configuration with strict mode and batching."""
    model = small_transformer_model  # For testing, use fixture.

    # --8<-- [start:generation-settings-config]
    from sieves.model_wrappers.utils import ModelSettings
    from sieves import tasks

    classifier = tasks.Classification(
        labels={
            "science": "Scientific topics and research",
            "politics": "Political news and government"
        },
        model=model,
        model_settings=ModelSettings(strict=True),
        batch_size=8,
    )
    # --8<-- [end:generation-settings-config]

    # Assertions for testing.
    assert classifier is not None
    assert classifier._model_settings.strict is True


def test_inference_mode_example(small_outlines_model):
    """Test model type-specific inference mode configuration."""
    model = small_outlines_model  # For testing, use fixture.

    # --8<-- [start:inference-mode-config]
    import outlines
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sieves.model_wrappers import outlines_
    from sieves.model_wrappers.utils import ModelSettings
    from sieves import tasks

    model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    model = outlines.models.from_transformers(
        AutoModelForCausalLM.from_pretrained(model_name),
        AutoTokenizer.from_pretrained(model_name)
    )
    classifier = tasks.Classification(
        labels=["science", "politics"],
        model=model,
        model_settings=ModelSettings(
            strict=True,
            inference_mode=outlines_.InferenceMode.json  # Specifies how to parse results.
        ),
        batch_size=8,
    )
    # --8<-- [end:inference-mode-config]

    # Assertions for testing.
    assert classifier is not None
    assert classifier._model_settings.inference_mode == outlines_.InferenceMode.json


def test_readme_quick_start_basic(small_outlines_model):
    """Test the Quick Start Classification example from README."""
    # Use fixture for actual test
    model = small_outlines_model

    # --8<-- [start:basic-classification]
    import outlines
    import transformers
    from sieves import Pipeline, tasks, Doc

    # Set up model.
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model = outlines.models.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained(model_name),
        transformers.AutoTokenizer.from_pretrained(model_name)
    )

    # Define task.
    task = tasks.Classification(labels=["science", "politics"], model=model)
    # Define pipeline with at least one task.
    pipeline = Pipeline(task)

    # Define documents to analyze.
    doc = Doc(text="The new telescope captures images of distant galaxies.")

    # Run pipeline a print results.
    docs = list(pipeline([doc]))
    print(docs[0].results) # {'Classification': [('science', 1.0), ('politics', 0.0)]}
    print(docs[0].meta)    # {'Classification': {'raw': ['{ "science": 1.0, "politics": 0'], 'usage': {'input_tokens': 2, 'output_tokens': 2, 'chunks': [{'input_tokens': 2, 'output_tokens': 2}]}}, 'usage': {'input_tokens': 2, 'output_tokens': 2}}
    # This produces: {'Classification': [('science', 1.0), ('politics', 0.0)]}
    # --8<-- [end:basic-classification]

    # Assertions for testing (not shown in docs).
    assert docs[0].results is not None
    # Verify it returned classification results
    assert "Classification" in docs[0].results
    assert docs[0].results["Classification"] is not None
    assert docs[0].meta["Classification"] is not None


@pytest.mark.slow
@pytest.mark.parametrize("runtime", [ModelType.dspy], indirect=True)
def test_readme_advanced_example(runtime):
    """Test the Advanced IE + PDF example from README."""
    model = runtime.model

    # --8<-- [start:readme-advanced]
    import dspy
    import os
    import pydantic
    import chonkie
    import tokenizers
    from sieves import tasks, Doc

    # Define which schema of entity to extract.
    class Equation(pydantic.BaseModel, frozen=True):
        id: str = pydantic.Field(description="ID/index of equation in paper.")
        equation: str = pydantic.Field(description="Equation as shown in paper.")

    # Setup DSPy model
    model = dspy.LM(
        "openrouter/google/gemini-3-flash-preview",
        api_base="https://openrouter.ai/api/v1/",
        api_key=os.environ["OPENROUTER_API_KEY"]
    )

    # Build Pipeline: Ingest -> Chunk -> Extract.
    pipeline = (
        tasks.Ingestion() +
        tasks.Chunking(chonkie.TokenChunker(tokenizers.Tokenizer.from_pretrained("gpt2"))) +
        tasks.InformationExtraction(entity_type=Equation, model=model)
    )

    # Define docs to analyze.
    doc = Doc(uri="https://arxiv.org/pdf/1204.0162")

    # Run pipeline.
    results = list(pipeline([doc]))

    # Print esults.
    for equation in results[0].results["InformationExtraction"]:
        print(equation)
    # --8<-- [end:readme-advanced]

    # Assertions for testing (not shown in docs)
    assert results[0].results is not None
    assert "InformationExtraction" in results[0].results
