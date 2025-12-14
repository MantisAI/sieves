"""
Test file containing examples for the Getting Started guide.

These code blocks are referenced in docs/guides/getting_started.md using snippet injection.

Usage in markdown:
    ```python
    --8<-- "sieves/tests/docs/test_getting_started.py:basic-classification"
    ```
"""

import pytest
import outlines
from sieves import Doc, Pipeline, tasks


def test_basic_classification_example(small_outlines_model):
    """Test the basic classification example from the getting started guide."""
    model = small_outlines_model  # For testing, use fixture

    # --8<-- [start:basic-classification]
    import outlines
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sieves import Pipeline, tasks, Doc

    # Create a document
    doc = Doc(text="Special relativity applies to all physical phenomena in the absence of gravity.")

    # Choose a model (using a small but capable model)
    model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    model = outlines.models.from_transformers(
        AutoModelForCausalLM.from_pretrained(model_name),
        AutoTokenizer.from_pretrained(model_name)
    )

    # Create and run the pipeline (verbose init)
    pipeline = Pipeline([
        tasks.predictive.Classification(
            labels=["science", "politics"],
            model=model,
        )
    ])

    # Print the classification result
    for doc in pipeline([doc]):
        print(doc.results)
    # --8<-- [end:basic-classification]

    # Assertions for testing (not shown in docs)
    assert doc.results is not None
    assert len(doc.results) > 0


def test_label_descriptions_example(small_outlines_model):
    """Test classification with label descriptions."""
    model = small_outlines_model  # For testing, use fixture

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

    # Use dict format to provide descriptions
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

    # From text
    doc = Doc(text="Your text here")
    # --8<-- [end:doc-from-text]
    assert doc.text == "Your text here"

    # --8<-- [start:doc-with-metadata]
    # With metadata
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

    # From a file (requires docling)
    doc = Doc(uri="path/to/your/file.pdf")
    # --8<-- [end:doc-from-uri]
    # Note: This would fail without a real file, hence the skip marker


def test_advanced_pipeline_example(example_chunker, small_outlines_model):
    """Test advanced pipeline with chunking and extraction."""
    # --8<-- [start:advanced-pipeline]
    import chonkie
    import outlines
    import tokenizers
    import pydantic
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sieves import Doc, Pipeline, tasks

    # Create a tokenizer for chunking
    tokenizer = tokenizers.Tokenizer.from_pretrained("bert-base-uncased")

    # Initialize components
    chunker = tasks.Chunking(
        chunker=chonkie.TokenChunker(tokenizer, chunk_size=512, chunk_overlap=50)
    )

    # Choose a model for information extraction
    model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    model = outlines.models.from_transformers(
        AutoModelForCausalLM.from_pretrained(model_name),
        AutoTokenizer.from_pretrained(model_name)
    )

    # Define the structure of information you want to extract
    class PersonInfo(pydantic.BaseModel):
        name: str
        age: int | None = None
        occupation: str | None = None

    # Create an information extraction task
    extractor = tasks.predictive.InformationExtraction(
        entity_type=PersonInfo,
        model=model,
    )

    # Create the pipeline (use + for succinct chaining)
    pipeline = chunker + extractor

    # Process a document
    doc = Doc(text="Marie Curie died at the age of 66 years.")
    results = list(pipeline([doc]))

    # Access the extracted information
    for result in results:
        print(result.results["InformationExtraction"])
    # --8<-- [end:advanced-pipeline]

    assert len(results) > 0


def test_generation_settings_example(small_transformer_model):
    """Test GenerationSettings configuration with strict mode and batching."""
    model = small_transformer_model  # For testing, use fixture

    # --8<-- [start:generation-settings-config]
    from sieves.engines.utils import GenerationSettings
    from sieves import tasks

    classifier = tasks.Classification(
        labels={
            "science": "Scientific topics and research",
            "politics": "Political news and government"
        },
        model=model,
        generation_settings=GenerationSettings(strict_mode=True),
        batch_size=8,
    )
    # --8<-- [end:generation-settings-config]

    # Assertions for testing
    assert classifier is not None
    assert classifier._generation_settings.strict_mode is True


def test_inference_mode_example(small_outlines_model):
    """Test engine-specific inference mode configuration."""
    model = small_outlines_model  # For testing, use fixture

    # --8<-- [start:inference-mode-config]
    import outlines
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sieves.engines import outlines_
    from sieves.engines.utils import GenerationSettings
    from sieves import tasks

    model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    model = outlines.models.from_transformers(
        AutoModelForCausalLM.from_pretrained(model_name),
        AutoTokenizer.from_pretrained(model_name)
    )
    classifier = tasks.Classification(
        labels=["science", "politics"],
        model=model,
        generation_settings=GenerationSettings(
            strict_mode=True,
            inference_mode=outlines_.InferenceMode.json  # Specifies how to parse results
        ),
        batch_size=8,
    )
    # --8<-- [end:inference-mode-config]

    # Assertions for testing
    assert classifier is not None
    assert classifier._generation_settings.inference_mode == outlines_.InferenceMode.json
