# mypy: ignore-errors
import pytest
import pydantic
from sieves import Doc, Pipeline
from sieves.model_wrappers import ModelType
from sieves.tasks import (
    Classification,
    InformationExtraction,
    NER,
    PIIMasking,
    QuestionAnswering,
    SentimentAnalysis,
    Summarization,
    Translation,
)

class Person(pydantic.BaseModel):
    """Pydantic model for testing information extraction."""
    model_config = pydantic.ConfigDict(frozen=True)
    name: str

@pytest.mark.parametrize(
    "task_cls, task_kwargs, doc_fixture",
    [
        (Classification, {"labels": ["science", "politics"]}, "classification_docs"),
        (InformationExtraction, {"entity_type": Person}, "information_extraction_docs"),
        (NER, {"entities": ["PER", "ORG"]}, "ner_docs"),
        (PIIMasking, {}, "pii_masking_docs"),
        (QuestionAnswering, {"questions": ["What is history?"]}, "qa_docs"),
        (SentimentAnalysis, {}, "sentiment_analysis_docs"),
        (Summarization, {"n_words": 10}, "summarization_docs"),
        (Translation, {"to": "German"}, "translation_docs"),
    ],
)
@pytest.mark.parametrize("include_meta", [True, False])
@pytest.mark.parametrize("runtime", [ModelType.dspy], indirect=["runtime"])
def test_include_meta_generic(task_cls, task_kwargs, doc_fixture, include_meta, runtime, request):
    """Test whether the include_meta flag correctly controls the population of doc.meta."""
    docs = request.getfixturevalue(doc_fixture)
    # Ensure docs are fresh.
    docs = [Doc(text=d.text) for d in docs]

    task_id = "test_task"
    task = task_cls(
        task_id=task_id,
        model=runtime.model,
        model_settings=runtime.model_settings,
        include_meta=include_meta,
        batch_size=runtime.batch_size,
        **task_kwargs
    )

    pipe = Pipeline(task)
    results = list(pipe(docs))

    for doc in results:
        if include_meta:
            assert task_id in doc.meta
            assert "raw" in doc.meta[task_id]
            assert isinstance(doc.meta[task_id]["raw"], list)
            assert len(doc.meta[task_id]["raw"]) > 0
        else:
            assert task_id not in doc.meta
