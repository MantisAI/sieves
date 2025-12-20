"""
Tests for task usage documentation snippets.
"""
from typing import Any
import pytest
import pydantic
import dspy
from sieves import Pipeline, tasks, Doc
from sieves.tasks.predictive.information_extraction import FewshotExampleMulti, FewshotExampleSingle

# Use a real model class but with dummy credentials.
# This satisfies type checks and initialization logic without making network calls.
model = dspy.LM("openai/gpt-4o-mini", api_key="dummy")

def test_classification_usage():
    # --8<-- [start:classification-list]
    from sieves import tasks

    task = tasks.Classification(
        labels=["positive", "negative"],
        model=model,
    )
    # --8<-- [end:classification-list]
    assert task

    # --8<-- [start:classification-dict]
    task = tasks.Classification(
        labels={
            "positive": "Positive sentiment regarding the subject.",
            "negative": "Negative sentiment regarding the subject.",
        },
        model=model,
    )
    # --8<-- [end:classification-dict]
    assert task

def test_ner_usage():
    # --8<-- [start:ner-usage]
    from sieves import tasks

    task = tasks.NER(
        entities=["PERSON", "ORGANIZATION", "LOCATION"],
        model=model,
    )
    # --8<-- [end:ner-usage]
    assert task

    # --8<-- [start:ner-dict-usage]
    task = tasks.NER(
        entities={
            "PERSON": "Names of people.",
            "ORGANIZATION": "Companies, agencies, institutions.",
        },
        model=model,
    )
    # --8<-- [end:ner-dict-usage]
    assert task

def test_information_extraction_usage():
    # --8<-- [start:ie-multi]
    import pydantic
    from sieves import tasks
    from sieves.tasks.predictive.information_extraction import FewshotExampleMulti

    class Person(pydantic.BaseModel, frozen=True):
        name: str
        age: int

    examples = [
        FewshotExampleMulti(
            text="Alice is 30 and Bob is 25.",
            entities=[Person(name="Alice", age=30), Person(name="Bob", age=25)]
        )
    ]

    task = tasks.InformationExtraction(
        entity_type=Person,
        mode="multi",
        fewshot_examples=examples,
        model=model,
    )
    # --8<-- [end:ie-multi]
    assert task

    # --8<-- [start:ie-single]
    from sieves import tasks
    from sieves.tasks.predictive.information_extraction import FewshotExampleSingle

    class Invoice(pydantic.BaseModel, frozen=True):
        id: str
        total: float

    examples = [
        FewshotExampleSingle(
            text="Invoice #123: $50.00",
            entity=Invoice(id="123", total=50.0)
        )
    ]

    task = tasks.InformationExtraction(
        entity_type=Invoice,
        mode="single",
        fewshot_examples=examples,
        model=model,
    )
    # --8<-- [end:ie-single]
    assert task

def test_sentiment_analysis_usage():
    # --8<-- [start:sentiment-usage]
    from sieves import tasks

    task = tasks.SentimentAnalysis(
        model=model,
    )
    # --8<-- [end:sentiment-usage]
    assert task

def test_question_answering_usage():
    # --8<-- [start:qa-usage]
    from sieves import tasks

    task = tasks.QuestionAnswering(
        questions=["What is the main topic?", "Who are the key figures?"],
        model=model,
    )
    # --8<-- [end:qa-usage]
    assert task

def test_summarization_usage():
    # --8<-- [start:summarization-usage]
    from sieves import tasks

    task = tasks.Summarization(
        model=model,
        n_words=100, # Guideline for summary length
    )
    # --8<-- [end:summarization-usage]
    assert task

def test_translation_usage():
    # --8<-- [start:translation-usage]
    from sieves import tasks

    task = tasks.Translation(
        to="Spanish",
        model=model,
    )
    # --8<-- [end:translation-usage]
    assert task

def test_pii_masking_usage():
    # --8<-- [start:pii-usage]
    from sieves import tasks

    task = tasks.PIIMasking(
        model=model,
    )
    # --8<-- [end:pii-usage]
    assert task

def test_chunking_usage():
    # --8<-- [start:chunking-usage]
    import chonkie
    from sieves import tasks

    # Define a chunker (e.g., using Chonkie)
    chunker_model = chonkie.RecursiveChunker()

    task = tasks.Chunking(
        chunker=chunker_model,
    )
    # --8<-- [end:chunking-usage]
    assert task

def test_ingestion_usage():
    # --8<-- [start:ingestion-usage]
    from sieves import tasks

    task = tasks.Ingestion(
        export_format="markdown",
    )
    # --8<-- [end:ingestion-usage]
    assert task
