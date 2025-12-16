"""
Shared fixtures for documentation example tests.

These fixtures provide commonly-used models and utilities for example tests,
loaded once per test session for efficiency.
"""

import pytest
import chonkie
import outlines
import tokenizers
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


@pytest.fixture(scope="session")
def small_transformer_model():
    """
    Small HuggingFace transformer model for fast zero-shot classification.

    Uses MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33 which is:
    - Fast (6 layers, 256 hidden size)
    - Accurate for zero-shot classification
    - No API key required
    """
    return transformers.pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33"
    )


@pytest.fixture(scope="session")
def example_tokenizer():
    """
    GPT-2 tokenizer for chunking examples.

    Used in chunking and preprocessing examples.
    """
    return tokenizers.Tokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="session")
def example_chunker(example_tokenizer):
    """
    Chonkie TokenChunker for document chunking examples.

    Configured with:
    - chunk_size=512 tokens
    - chunk_overlap=50 tokens

    Uses GPT-2 tokenizer.
    """
    return chonkie.TokenChunker(
        example_tokenizer,
        chunk_size=512,
        chunk_overlap=50
    )


@pytest.fixture(scope="session")
def small_outlines_model():
    """
    Small Outlines model for information extraction and structured generation.

    Uses HuggingFaceTB/SmolLM-135M-Instruct which is:
    - Very small (135M parameters)
    - Fast inference
    - Supports structured generation
    - No API key required
    """
    model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    return outlines.models.from_transformers(
        AutoModelForCausalLM.from_pretrained(model_name),
        AutoTokenizer.from_pretrained(model_name)
    )


@pytest.fixture(scope="session")
def small_dspy_model():
    """
    Small DSPy model for optimization and distillation examples.

    Uses OpenRouter with a small, fast model.
    Requires OPENROUTER_API_KEY environment variable.
    """
    import os
    import dspy

    openrouter_api_base = "https://openrouter.ai/api/v1/"
    openrouter_model_id = "google/gemini-2.5-flash-lite-preview-09-2025"

    return dspy.LM(
        f"openrouter/{openrouter_model_id}",
        api_base=openrouter_api_base,
        api_key=os.environ['OPENROUTER_API_KEY'],
        cache=False
    )
