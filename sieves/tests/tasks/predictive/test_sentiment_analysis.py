# mypy: ignore-errors
import pytest

from sieves import Doc, Pipeline
from sieves.model_wrappers import ModelType, ModelSettings, dspy_, langchain_, outlines_
from sieves.serialization import Config
from sieves.tasks import PredictiveTask, SentimentAnalysis
from sieves.tasks.predictive import sentiment_analysis


@pytest.mark.parametrize(
    "batch_runtime",
    SentimentAnalysis.supports(),
    indirect=["batch_runtime"],
)
@pytest.mark.parametrize("fewshot", [True, False])
def test_run(sentiment_analysis_docs, batch_runtime, fewshot):
    fewshot_examples = [
        sentiment_analysis.FewshotExample(
            text="The food was perfect, the service only ok.",
            sentiment_per_aspect={"food": 1.0, "service": 0.5, "overall": 0.8},
        ),
        sentiment_analysis.FewshotExample(
            text="The service was amazing - they take excellent care of their customers. The food was despicable "
            "though, I strongly recommend not to go.",
            sentiment_per_aspect={"food": 0.1, "service": 1.0, "overall": 0.3},
        ),
    ]

    fewshot_args = {"fewshot_examples": fewshot_examples} if fewshot else {}
    task = sentiment_analysis.SentimentAnalysis(
        task_id="sentiment_analysis",
        aspects=("food", "service"),
        model=batch_runtime.model,
        model_settings=batch_runtime.model_settings,
        batch_size=batch_runtime.batch_size,
        **fewshot_args,
    )
    pipe = Pipeline(task)
    docs = list(pipe(sentiment_analysis_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert doc.results["sentiment_analysis"]
        assert "sentiment_analysis" in doc.results
        assert "sentiment_analysis" in doc.meta
        assert "raw" in doc.meta["sentiment_analysis"]
        assert "usage" in doc.meta["sentiment_analysis"]
        assert "usage" in doc.meta

        print(f"Output: {doc.results['sentiment_analysis']}")
        print(f"Raw output: {doc.meta['sentiment_analysis']['raw']}")
        print(f"Usage: {doc.meta['sentiment_analysis']['usage']}")
        print(f"Total Usage: {doc.meta['usage']}")

    with pytest.raises(NotImplementedError):
        pipe["sentiment_analysis"].distill(None, None, None, None, None, None, None, None)

    if fewshot:
        _to_hf_dataset(task, docs)


def _to_hf_dataset(task: SentimentAnalysis, docs: list[Doc]) -> None:
    """Tests whether conversion to HF dataset works as expected.

    :param task: SentimentAnalysis task instance.
    :param docs: List of documents to convert.
    """
    dataset = task.to_hf_dataset(docs)
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


@pytest.mark.parametrize("batch_runtime", [ModelType.dspy], indirect=["batch_runtime"])
def test_serialization(dummy_docs, batch_runtime) -> None:
    pipe = Pipeline(
        [
            sentiment_analysis.SentimentAnalysis(
                task_id="sentiment_analysis",
                aspects=("food", "service"),
                model=batch_runtime.model,
                model_settings=batch_runtime.model_settings,
                batch_size=batch_runtime.batch_size,
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
                      'batch_size': {'is_placeholder': False, "value": -1},
                      'model_settings': {'is_placeholder': False,
                                              'value': {
                                                        'config_kwargs': None,
                                                        'inference_kwargs': None,
                                                        'init_kwargs': None,
                                                        'strict': True, 'inference_mode': None}},
                      'include_meta': {'is_placeholder': False, 'value': True},
                      'model': {'is_placeholder': True,
                                'value': 'dspy.clients.lm.LM'},
                      'prompt_instructions': {'is_placeholder': False,
                                          'value': None},
                      'task_id': {'is_placeholder': False,
                                  'value': 'sentiment_analysis'},
                      'condition': {'is_placeholder': False, 'value': None},
                      'version': Config.get_version()}]},
 'use_cache': {'is_placeholder': False, 'value': True},
 'version': Config.get_version()}

    Pipeline.deserialize(config=config, tasks_kwargs=[{"model": batch_runtime.model}])


@pytest.mark.parametrize(
    "batch_runtime",
    SentimentAnalysis.supports(),
    indirect=["batch_runtime"],
)
def test_inference_mode_override(batch_runtime) -> None:
    """Test that inference_mode parameter overrides the default value."""
    dummy = "dummy_inference_mode"

    task = sentiment_analysis.SentimentAnalysis(
        task_id="sentiment_analysis",
        aspects=("food", "service"),
        model=batch_runtime.model,
        model_settings=ModelSettings(inference_mode=dummy),
        batch_size=batch_runtime.batch_size,
    )

    assert task._bridge.inference_mode == dummy
