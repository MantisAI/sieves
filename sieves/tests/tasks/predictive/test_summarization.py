# mypy: ignore-errors
import pytest

from sieves import Doc, Pipeline
from sieves.model_wrappers import ModelType, ModelSettings, dspy_, langchain_, outlines_
from sieves.serialization import Config
from sieves.tasks import PredictiveTask, Summarization
from sieves.tasks.predictive import summarization
from sieves.tasks.predictive.schemas.summarization import Result


@pytest.mark.parametrize(
    "batch_runtime",
    Summarization.supports(),
    indirect=["batch_runtime"],
)
@pytest.mark.parametrize("fewshot", [True, False])
def test_run(summarization_docs, batch_runtime, fewshot) -> None:
    fewshot_examples = [
        summarization.FewshotExample(
            text="They counted: one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, "
            "fourteen.",
            n_words=6,
            summary="They counted from one to fourteen.",
            score=1.0,
        ),
        summarization.FewshotExample(
            text="Next in order were the Boeotians, led by Peneleos, Leitus, Arcesilaus, Prothoenor, and Clonius. "
            "These had with them fifty ships, and on board of each were a hundred and twenty young men of the "
            "Boeotians. Then came the men of Orchomenus, who lived in the realm of the Minyans, led by Ascalaphus"
            " and Ialmenus, sons of Mars. In their command were thirty ships. Next were the Phocians, led by"
            " Schedius and Epistrophus, sons of Iphitus the son of Naubolus. These had forty ships…",
            n_words=10,
            summary="Boeotians, Orchomenians, and Phocians sailed to Troy with many ships.",
            score=1.0,
        ),
    ]

    fewshot_args = {"fewshot_examples": fewshot_examples} if fewshot else {}
    task = summarization.Summarization(
        n_words=10,
        model=batch_runtime.model,
        model_settings=batch_runtime.model_settings,
        batch_size=batch_runtime.batch_size,
        **fewshot_args,
    )
    pipe = Pipeline(task)
    docs = list(pipe(summarization_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert "Summarization" in doc.results

        # Verify unified result types.
        assert isinstance(doc.results["Summarization"], summarization.Result)

        assert "Summarization" in doc.meta
        assert "raw" in doc.meta["Summarization"]
        assert "usage" in doc.meta["Summarization"]
        assert "usage" in doc.meta

        print(f"Output: {doc.results['Summarization']}")
        print(f"Raw output: {doc.meta['Summarization']['raw']}")
        print(f"Usage: {doc.meta['Summarization']['usage']}")
        print(f"Total Usage: {doc.meta['usage']}")

    with pytest.raises(NotImplementedError):
        pipe["Summarization"].distill(None, None, None, None, None, None, None, None)

    if fewshot:
        _to_hf_dataset(task, docs)


def _to_hf_dataset(task: Summarization, docs: list[Doc]) -> None:
    """Tests whether conversion to HF dataset works as expected.

    :param task: Summarization task instance.
    :param docs: List of documents to convert.
    """
    dataset = task.to_hf_dataset(docs)
    assert all([key in dataset.features for key in ("text", "summary")])
    assert len(dataset) == 2
    records = list(dataset)
    assert records[0]["text"].strip().startswith("The decay spreads over the State")
    assert records[1]["text"].strip().startswith("After all, the practical reason")
    for record in records:
        assert isinstance(record["summary"], str)

    with pytest.raises(KeyError):
        task.to_hf_dataset([Doc(text="This is a dummy text.")])


@pytest.mark.parametrize("batch_runtime", [ModelType.dspy], indirect=["batch_runtime"])
def test_serialization(summarization_docs, batch_runtime) -> None:
    pipe = Pipeline([
        summarization.Summarization(
            n_words=10,
            model=batch_runtime.model,
            model_settings=batch_runtime.model_settings,
            batch_size=batch_runtime.batch_size,
        )
    ])

    config = pipe.serialize()
    assert config.model_dump() == {'cls_name': 'sieves.pipeline.core.Pipeline',
 'tasks': {'is_placeholder': False,
           'value': [{'cls_name': 'sieves.tasks.predictive.summarization.core.Summarization',
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
                      'n_words': {'is_placeholder': False, 'value': 10},
                      'prompt_instructions': {'is_placeholder': False,
                                          'value': None},
                      'task_id': {'is_placeholder': False,
                                  'value': 'Summarization'},
                      'condition': {'is_placeholder': False, 'value': None},
                      'version': Config.get_version()}]},
 'use_cache': {'is_placeholder': False, 'value': True},
 'version': Config.get_version()}

    Pipeline.deserialize(config=config, tasks_kwargs=[{"model": batch_runtime.model}])


@pytest.mark.parametrize(
    "batch_runtime",
    Summarization.supports(),
    indirect=["batch_runtime"],
)
def test_inference_mode_override(batch_runtime) -> None:
    """Test that inference_mode parameter overrides the default value."""
    dummy = "dummy_inference_mode"

    task = summarization.Summarization(
        n_words=10,
        model=batch_runtime.model,
        model_settings=ModelSettings(inference_mode=dummy),
        batch_size=batch_runtime.batch_size,
    )

    assert task._bridge.inference_mode == dummy


@pytest.mark.parametrize("batch_runtime", [ModelType.dspy], indirect=["batch_runtime"])
def test_evaluation(batch_runtime) -> None:
    """Test evaluation for summarization using a real DSPy judge."""
    task = summarization.Summarization(n_words=10, model=batch_runtime.model, task_id="sum")

    # 1. Full overlap
    doc_full = Doc(text="The quick brown fox jumps over the lazy dog.")
    doc_full.results["sum"] = Result(summary="Fast fox jumps dog.", score=1.0)
    doc_full.gold["sum"] = Result(summary="Fast fox jumps dog.", score=1.0)
    report_full = task.evaluate([doc_full], judge=batch_runtime.model)
    assert report_full.metrics[task.metric] > 0.8

    # 2. No overlap
    doc_none = Doc(text="The quick brown fox jumps over the lazy dog.")
    doc_none.results["sum"] = Result(summary="The weather is nice today.", score=1.0)
    doc_none.gold["sum"] = Result(summary="Fast fox jumps dog.", score=1.0)
    report_none = task.evaluate([doc_none], judge=batch_runtime.model)
    assert report_none.metrics[task.metric] < 0.6

    # 3. Partial overlap
    doc_partial = Doc(text="The quick brown fox jumps over the lazy dog.")
    doc_partial.results["sum"] = Result(summary="A fox jumps.", score=1.0)
    doc_partial.gold["sum"] = Result(summary="The quick brown fox jumps over the lazy dog.", score=1.0)
    report_partial = task.evaluate([doc_partial], judge=batch_runtime.model)
    assert 0.2 <= report_partial.metrics[task.metric] <= 0.8
