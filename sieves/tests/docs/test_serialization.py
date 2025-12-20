"""
Test file containing examples for the Serialization guide.

These code blocks are referenced in docs/guides/serialization.md using snippet injection.
"""

import pytest
from pathlib import Path


def test_basic_serialization(small_outlines_model, tmp_path):
    """Test basic pipeline serialization and loading."""
    model = small_outlines_model

    # --8<-- [start:serialization-basic-pipeline]
    import outlines
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sieves import Pipeline, tasks, Doc
    from pathlib import Path

    # Create a basic classification pipeline
    model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    model = outlines.models.from_transformers(
        AutoModelForCausalLM.from_pretrained(model_name),
        AutoTokenizer.from_pretrained(model_name)
    )
    classifier = tasks.predictive.Classification(labels=["science", "politics"], model=model)
    pipeline = Pipeline([classifier])

    # Save the pipeline configuration
    config_path = Path("classification_pipeline.yml")
    pipeline.dump(config_path)

    # Load the pipeline configuration
    loaded_pipeline = Pipeline.load(config_path, [{"model": outlines.models.from_transformers(
        AutoModelForCausalLM.from_pretrained(model_name),
        AutoTokenizer.from_pretrained(model_name)
    )}])

    # Use the loaded pipeline
    doc = Doc(text="Special relativity applies to all physical phenomena in the absence of gravity.")
    results = list(loaded_pipeline([doc]))
    print(results[0].results["Classification"])
    # --8<-- [end:serialization-basic-pipeline]

    # Assertions
    assert results[0].results["Classification"] is not None

    # Cleanup
    config_path.unlink()


def test_complex_serialization(example_tokenizer, small_outlines_model, tmp_path):
    """Test serialization with complex third-party objects."""
    tokenizer = example_tokenizer
    model = small_outlines_model

    # --8<-- [start:serialization-complex-pipeline]
    # --8<-- [start:serialization-complex-setup]
    import chonkie
    import tokenizers
    import outlines
    import pydantic
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sieves import Pipeline, tasks

    # Create a tokenizer for chunking
    tokenizer = tokenizers.Tokenizer.from_pretrained("bert-base-uncased")
    chonkie_chunker = chonkie.TokenChunker(tokenizer, chunk_size=512, chunk_overlap=50)
    chunker = tasks.Chunking(chunker=chonkie_chunker)

    model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    model = outlines.models.from_transformers(
        AutoModelForCausalLM.from_pretrained(model_name),
        AutoTokenizer.from_pretrained(model_name)
    )
    # --8<-- [end:serialization-complex-setup]


    # --8<-- [start:serialization-complex-entity-task]
    class PersonInfo(pydantic.BaseModel, frozen=True):
        name: str
        age: int | None = None
        occupation: str | None = None


    extractor = tasks.predictive.InformationExtraction(entity_type=PersonInfo, model=model)
    # --8<-- [end:serialization-complex-entity-task]

    # --8<-- [start:serialization-complex-save]
    # Create and save the pipeline
    pipeline = chunker + extractor
    pipeline.dump("extraction_pipeline.yml")
    # --8<-- [end:serialization-complex-save]

    # --8<-- [start:serialization-complex-load]
    # Load the pipeline with initialization parameters for each task
    loaded_pipeline = Pipeline.load(
        "extraction_pipeline.yml",
        [
            {"chunker": chonkie_chunker},
            {
                "entity_type": PersonInfo,
                "model": model
            },
        ]
    )
    # --8<-- [end:serialization-complex-load]
    # --8<-- [end:serialization-complex-pipeline]

    # Assertions
    assert loaded_pipeline is not None
    assert len(loaded_pipeline.tasks) == 2

    # Cleanup
    Path("extraction_pipeline.yml").unlink()
