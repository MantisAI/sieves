"""
Tests for preprocessing task usage documentation snippets.
"""
import pytest
from sieves import tasks

def test_docling_usage():
    pytest.importorskip("docling")
    # --8<-- [start:docling-usage]
    from sieves.tasks.preprocessing.ingestion.docling_ import Docling

    # Basic usage
    task = Docling(export_format="markdown")
    # --8<-- [end:docling-usage]
    assert task

def test_chonkie_usage():
    try:
        import tokenizers
        tokenizers.Tokenizer.from_pretrained("gpt2")
    except Exception:
        pytest.skip("tokenizers not ready")

    # --8<-- [start:chonkie-usage]
    import chonkie
    import tokenizers
    from sieves.tasks.preprocessing.chunking.chonkie_ import Chonkie

    # Setup tokenizer and chunker
    tokenizer = tokenizers.Tokenizer.from_pretrained("gpt2")
    chunker = chonkie.TokenChunker(tokenizer)

    task = Chonkie(chunker=chunker)
    # --8<-- [end:chonkie-usage]
    assert task

def test_naive_usage():
    # --8<-- [start:naive-usage]
    from sieves.tasks.preprocessing.chunking.naive import NaiveChunker

    # Chunk by sentence count interval
    task = NaiveChunker(interval=5)
    # --8<-- [end:naive-usage]
    assert task
