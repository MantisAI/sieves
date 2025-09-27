# mypy: ignore-errors
import pytest

from sieves import Doc, Pipeline
from sieves.engines import EngineType
from sieves.serialization import Config
from sieves.tasks import PredictiveTask
from sieves.tasks.predictive import sentiment_analysis


@pytest.mark.parametrize(
    "batch_runtime",
    (
        EngineType.dspy,
        EngineType.instructor,
        EngineType.langchain,
        EngineType.ollama,
        EngineType.outlines,
        # EngineType.vllm,
    ),
    indirect=["batch_runtime"],
)
@pytest.mark.parametrize("fewshot", [True, False])
def test_run(sentiment_analysis_docs, batch_runtime, fewshot):
    fewshot_examples = [
        sentiment_analysis.FewshotExample(
            text="The food was perfect, the service only ok.",
            reasoning="The text is very positive about the quality of the food, and neutral about the service quality."
            " The overall sentiment is hence positive.",
            sentiment_per_aspect={"food": 1.0, "service": 0.5, "overall": 0.8},
        ),
        sentiment_analysis.FewshotExample(
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
                task_id="sentiment_analysis",
                aspects=("food", "service"),
                model=batch_runtime.model,
                generation_settings=batch_runtime.generation_settings,
                **fewshot_args,
            ),
        ]
    )
    docs = list(pipe(sentiment_analysis_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert doc.results["sentiment_analysis"]
        assert "sentiment_analysis" in doc.results

    with pytest.raises(NotImplementedError):
        pipe["sentiment_analysis"].distill(None, None, None, None, None, None, None, None)


@pytest.mark.parametrize("batch_runtime", [EngineType.dspy], indirect=["batch_runtime"])
def test_to_hf_dataset(dummy_docs, batch_runtime) -> None:
    task = sentiment_analysis.SentimentAnalysis(
        task_id="sentiment_analysis",
        aspects=("food", "service"),
        model=batch_runtime.model,
        generation_settings=batch_runtime.generation_settings,
    )

    assert isinstance(task, PredictiveTask)
    dataset = task.to_hf_dataset(task(dummy_docs))
    assert all([key in dataset.features for key in ("text", "aspect")])
    assert len(dataset) == 2
    dataset_records = list(dataset)
    for rec in dataset_records:
        assert isinstance(rec["aspect"], list)
        for v in rec["aspect"]:
            assert isinstance(v, float)
        assert isinstance(rec["text"], str)

    with pytest.raises(KeyError):
        task.to_hf_dataset([Doc(text="This is a dummy text.")])


@pytest.mark.parametrize("batch_runtime", [EngineType.dspy], indirect=["batch_runtime"])
def test_serialization(dummy_docs, batch_runtime) -> None:
    pipe = Pipeline(
        [
            sentiment_analysis.SentimentAnalysis(
                task_id="sentiment_analysis",
                aspects=("food", "service"),
                model=batch_runtime.model,
                generation_settings=batch_runtime.generation_settings,
            )
        ]
    )

    config = pipe.serialize()
    assert config.model_dump() == {'cls_name': 'sieves.pipeline.core.Pipeline',
 'tasks': {'is_placeholder': False,
           'value': [{'aspects': {'is_placeholder': False,
                                  'value': ('food', 'overall', 'service')},
                      'cls_name': 'sieves.tasks.predictive.sentiment_analysis.core.SentimentAnalysis',
                      'fewshot_examples': {'is_placeholder': False,
                                           'value': ()},
                      'generation_settings': {'is_placeholder': False,
                                              'value': {'batch_size': -1,
                                                        'config_kwargs': None,
                                                        'inference_kwargs': None,
                                                        'init_kwargs': None,
                                                        'strict_mode': False}},
                      'include_meta': {'is_placeholder': False, 'value': True},
                      'model': {'is_placeholder': True,
                                'value': 'dspy.clients.lm.LM'},
                      'prompt_signature_desc': {'is_placeholder': False,
                                                'value': None},
                      'prompt_template': {'is_placeholder': False,
                                          'value': None},
                      'task_id': {'is_placeholder': False,
                                  'value': 'sentiment_analysis'},
                      'version': Config.get_version()}]},
 'use_cache': {'is_placeholder': False, 'value': True},
 'version': Config.get_version()}

    Pipeline.deserialize(config=config, tasks_kwargs=[{"model": batch_runtime.model}])
