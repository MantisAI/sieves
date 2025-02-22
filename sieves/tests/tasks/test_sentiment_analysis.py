# mypy: ignore-errors
import pytest

from sieves import Doc, Pipeline
from sieves.engines import EngineType
from sieves.tasks import PredictiveTask
from sieves.tasks.predictive import sentiment_analysis


@pytest.mark.parametrize(
    "batch_engine",
    (EngineType.dspy, EngineType.instructor, EngineType.langchain, EngineType.ollama, EngineType.outlines),
    indirect=["batch_engine"],
)
@pytest.mark.parametrize("fewshot", [True, False])
def test_run(sentiment_analysis_docs, batch_engine, fewshot):
    fewshot_examples = [
        sentiment_analysis.TaskFewshotExample(
            text="The food was perfect, the service only ok.",
            reasoning="The text is very positive about the quality of the food, and neutral about the service quality."
            " The overall sentiment is hence positive.",
            sentiment_per_aspect={"food": 1.0, "service": 0.5, "overall": 0.8},
        ),
        sentiment_analysis.TaskFewshotExample(
            text="The service was amazing - they take excellent care of their customers. The food was despicable "
            "though, I strongly recommend not to go.",
            reasoning="While the service is judged as amazing, hence very positive, the assessment of the food is very "
            "negative. The overall sentiment is strongly negative.",
            sentiment_per_aspect={"food": 0.1, "service": 1.0, "overall": 0.3},
        ),
    ]

    fewshot_args = {"fewshot_examples": fewshot_examples} if fewshot else {}
    pipe = Pipeline(
        [
            sentiment_analysis.SentimentAnalysis(
                task_id="sentiment_analysis", aspects=("food", "service"), engine=batch_engine, **fewshot_args
            ),
        ]
    )
    docs = list(pipe(sentiment_analysis_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert doc.results["sentiment_analysis"]
        assert "sentiment_analysis" in doc.results


@pytest.mark.parametrize("batch_engine", [EngineType.outlines], indirect=["batch_engine"])
def test_to_dataset(dummy_docs, batch_engine) -> None:
    task = sentiment_analysis.SentimentAnalysis(
        task_id="sentiment_analysis", aspects=("food", "service"), engine=batch_engine
    )

    assert isinstance(task, PredictiveTask)
    dataset = task.to_dataset(task(dummy_docs))
    assert all([key in dataset.features for key in ("text", "aspect")])
    assert len(dataset) == 2
    dataset_records = list(dataset)
    for rec in dataset_records:
        assert isinstance(rec["aspect"], list)
        for v in rec["aspect"]:
            assert isinstance(v, float)
        assert isinstance(rec["text"], str)

    with pytest.raises(KeyError):
        task.to_dataset([Doc(text="This is a dummy text.")])


@pytest.mark.parametrize("batch_engine", [EngineType.outlines], indirect=["batch_engine"])
def test_serialization(dummy_docs, batch_engine) -> None:
    pipe = Pipeline(
        [
            sentiment_analysis.SentimentAnalysis(
                task_id="sentiment_analysis", aspects=("food", "service"), engine=batch_engine
            )
        ]
    )
    list(pipe(dummy_docs))

    config = pipe.serialize()
    assert config.model_dump() == {
        "cls_name": "sieves.pipeline.core.Pipeline",
        "tasks": {
            "is_placeholder": False,
            "value": [
                {
                    "aspects": {"is_placeholder": False, "value": ("food", "overall", "service")},
                    "cls_name": "sieves.tasks.predictive.sentiment_analysis.core.SentimentAnalysis",
                    "engine": {
                        "is_placeholder": False,
                        "value": {
                            "cls_name": "sieves.engines.outlines_.Outlines",
                            "inference_kwargs": {"is_placeholder": False, "value": {}},
                            "init_kwargs": {"is_placeholder": False, "value": {}},
                            "model": {"is_placeholder": True, "value": "outlines.models.transformers.Transformers"},
                            "version": "0.7.0",
                        },
                    },
                    "fewshot_examples": {"is_placeholder": False, "value": ()},
                    "include_meta": {"is_placeholder": False, "value": True},
                    "prompt_signature_desc": {"is_placeholder": False, "value": None},
                    "prompt_template": {"is_placeholder": False, "value": None},
                    "show_progress": {"is_placeholder": False, "value": True},
                    "task_id": {"is_placeholder": False, "value": "sentiment_analysis"},
                    "version": "0.7.0",
                }
            ],
        },
        "version": "0.7.0",
    }

    Pipeline.deserialize(config=config, tasks_kwargs=[{"engine": {"model": batch_engine.model}}])
