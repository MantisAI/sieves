"""
Tests for model wrapper usage documentation snippets.
"""
import os
import pytest
from sieves import tasks

def test_dspy_usage():
    # --8<-- [start:dspy-usage]
    import dspy
    from sieves import tasks

    # Initialize a DSPy Language Model
    model = dspy.LM("openai/gpt-4o-mini", api_key="dummy")

    # Pass it to a task
    task = tasks.SentimentAnalysis(model=model)
    # --8<-- [end:dspy-usage]
    assert task

def test_outlines_usage():
    # --8<-- [start:outlines-usage]
    import outlines
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sieves import tasks

    # Initialize an Outlines model
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model = outlines.models.from_transformers(
        AutoModelForCausalLM.from_pretrained(model_name),
        AutoTokenizer.from_pretrained(model_name)
    )

    # Pass it to a task
    task = tasks.SentimentAnalysis(model=model)
    # --8<-- [end:outlines-usage]
    assert task

def test_langchain_usage():
    # --8<-- [start:langchain-usage]
    from langchain_openai import ChatOpenAI
    from sieves import tasks

    # Initialize a LangChain Chat Model
    model = ChatOpenAI(model="gpt-4o-mini", api_key="dummy")

    # Pass it to a task
    task = tasks.SentimentAnalysis(model=model)
    # --8<-- [end:langchain-usage]
    assert task

def test_gliner_usage():
    # --8<-- [start:gliner-usage]
    import gliner2
    from sieves import tasks

    # Initialize a GLiNER model
    model = gliner2.GLiNER2.from_pretrained("fastino/gliner2-base-v1")

    # Pass it to a task (Note: GLiNER is specialized for extraction/classification)
    task = tasks.NER(
        entities=["PERSON", "LOC"],
        model=model
    )
    # --8<-- [end:gliner-usage]
    assert task

def test_huggingface_usage():
    # --8<-- [start:huggingface-usage]
    import transformers
    from sieves import tasks

    # Initialize a Hugging Face pipeline
    model = transformers.pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33"
    )

    # Pass it to a task
    task = tasks.Classification(
        labels=["positive", "negative"],
        model=model
    )
    # --8<-- [end:huggingface-usage]
    assert task
