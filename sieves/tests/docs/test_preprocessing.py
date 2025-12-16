"""
Test file containing examples for the Preprocessing guide.

These code blocks are referenced in docs/guides/preprocessing.md using snippet injection.

Usage in markdown:
    ```python
    --8<-- "sieves/tests/docs/test_preprocessing.py:ingestion-basic"
    ```
"""

import pytest


def test_basic_ingestion_example():
    """Test basic document ingestion/parsing example."""
    # --8<-- [start:ingestion-basic]
    from sieves import Pipeline, tasks, Doc

    # Create a document parser
    parser = tasks.preprocessing.Ingestion()

    # Create a pipeline with the parser
    pipeline = Pipeline([parser])

    # Process documents (requires actual PDF/DOCX files)
    docs = [
        Doc(uri="path/to/document.pdf"),
        Doc(uri="path/to/another.docx")
    ]
    # Note: Ingestion requires actual files and optional dependencies
    # Install with: pip install "sieves[ingestion]"
    # --8<-- [end:ingestion-basic]

    # Verify setup is correct
    assert parser is not None
    assert pipeline is not None
    assert len(docs) == 2


def test_custom_converter_example():
    """Test ingestion with custom converter and export format."""
    # --8<-- [start:ingestion-custom-converter]
    from sieves import Pipeline, tasks, Doc

    # Create a document parser with custom export format
    parser = tasks.preprocessing.Ingestion(export_format="html")

    # Create a pipeline with the parser
    pipeline = Pipeline([parser])

    # Process documents (requires actual PDF/DOCX files)
    docs = [
        Doc(uri="path/to/document.pdf"),
        Doc(uri="path/to/another.docx")
    ]
    # Note: Ingestion requires actual files and optional dependencies
    # Install with: pip install "sieves[ingestion]"
    # --8<-- [end:ingestion-custom-converter]

    assert parser is not None
    assert pipeline is not None
    assert len(docs) == 2


def test_chunking_example(example_tokenizer):
    """Test Chunking task with chonkie library."""
    tokenizer = example_tokenizer  # For testing, use fixture

    # --8<-- [start:chunking-chonkie-basic]
    import chonkie
    import tokenizers
    from sieves import Pipeline, tasks, Doc

    # Create a tokenizer for chunking
    tokenizer = tokenizers.Tokenizer.from_pretrained("bert-base-uncased")

    # Create a token-based chunker
    chunker = tasks.Chunking(
        chunker=chonkie.TokenChunker(tokenizer, chunk_size=512, chunk_overlap=50)
    )

    # Create and run the pipeline
    pipeline = Pipeline(chunker)
    doc = Doc(text="Your long document text here...")
    chunked_docs = list(pipeline([doc]))

    # Access the chunks
    for chunk in chunked_docs[0].chunks:
        print(f"Chunk: {chunk}")
    # --8<-- [end:chunking-chonkie-basic]

    # Assertions for testing
    assert len(chunked_docs) > 0
    assert chunked_docs[0].chunks is not None


def test_combined_preprocessing_pipeline(example_tokenizer):
    """Test combined preprocessing pipeline with chunking."""
    tokenizer = example_tokenizer  # For testing, use fixture

    # --8<-- [start:preprocessing-combined-pipeline]
    from sieves import tasks, Doc, Pipeline
    import chonkie
    import tokenizers

    # Create a tokenizer and chunker
    tokenizer = tokenizers.Tokenizer.from_pretrained("bert-base-uncased")
    chunker = tasks.Chunking(
        chunker=chonkie.TokenChunker(tokenizer, chunk_size=512, chunk_overlap=50)
    )

    # Create a pipeline
    pipeline = Pipeline(chunker)

    # Process a document with text
    doc = Doc(text="This is a long document that will be split into chunks. " * 100)
    processed_doc = list(pipeline([doc]))[0]

    # Access the chunks
    print(f"Number of chunks: {len(processed_doc.chunks)}")
    for i, chunk in enumerate(processed_doc.chunks):
        print(f"Chunk {i}: {chunk[:100]}...")  # Print first 100 chars of each chunk
    # --8<-- [end:preprocessing-combined-pipeline]

    assert processed_doc.chunks is not None
    assert len(processed_doc.chunks) > 0


def test_metadata_inclusion_example():
    """Test metadata inclusion in preprocessing tasks."""
    # --8<-- [start:metadata-inclusion]
    from sieves import tasks

    parser = tasks.preprocessing.Ingestion(include_meta=True)
    # --8<-- [end:metadata-inclusion]

    # Assertion for testing
    assert parser is not None


def test_metadata_access_example():
    """Demonstrate how to access metadata from processed documents."""
    # Create dummy data to demonstrate the pattern
    from sieves import Doc

    # Simulate processed docs with metadata
    processed_docs = [
        Doc(text="example", meta={"Ingestion": {"status": "ok"}, "Chunker": {"num_chunks": 3}})
    ]

    # --8<-- [start:metadata-access]
    doc = processed_docs[0]
    print(doc.meta["Ingestion"])  # Access parser metadata
    print(doc.meta["Chunker"])  # Access chunker metadata
    # --8<-- [end:metadata-access]

    assert doc.meta is not None
