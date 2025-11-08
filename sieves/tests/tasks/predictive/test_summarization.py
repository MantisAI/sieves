# mypy: ignore-errors
import pytest

from sieves import Doc, Pipeline
from sieves.engines import EngineType, GenerationSettings, dspy_, langchain_, outlines_
from sieves.serialization import Config
from sieves.tasks import PredictiveTask
from sieves.tasks.predictive import summarization


@pytest.mark.parametrize(
    "batch_runtime",
    (
        EngineType.dspy,
        EngineType.glix,
        EngineType.langchain,
        EngineType.outlines,
    ),
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
        ),
        summarization.FewshotExample(
            text="Next in order were the Boeotians, led by Peneleos, Leitus, Arcesilaus, Prothoenor, and Clonius. "
            "These had with them fifty ships, and on board of each were a hundred and twenty young men of the "
            "Boeotians. Then came the men of Orchomenus, who lived in the realm of the Minyans, led by Ascalaphus"
            " and Ialmenus, sons of Mars. In their command were thirty ships. Next were the Phocians, led by"
            " Schedius and Epistrophus, sons of Iphitus the son of Naubolus. These had forty ships…",
            n_words=10,
            summary="Boeotians, Orchomenians, and Phocians sailed to Troy with many ships.",
        ),
    ]

    fewshot_args = {"fewshot_examples": fewshot_examples} if fewshot else {}
    pipe = Pipeline([
        summarization.Summarization(
            n_words=10,
            model=batch_runtime.model,
            generation_settings=batch_runtime.generation_settings,
            batch_size=batch_runtime.batch_size,
            **fewshot_args,
        )
    ])
    docs = list(pipe(summarization_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert "Summarization" in doc.results

    with pytest.raises(NotImplementedError):
        pipe["Summarization"].distill(None, None, None, None, None, None, None, None)


@pytest.mark.parametrize("batch_runtime", [EngineType.dspy], indirect=["batch_runtime"])
def test_to_hf_dataset(summarization_docs, batch_runtime) -> None:
    task = summarization.Summarization(
        n_words=10,
        model=batch_runtime.model,
        generation_settings=batch_runtime.generation_settings,
        batch_size=batch_runtime.batch_size,
    )
    pipe = Pipeline(task)
    docs = pipe(summarization_docs)

    assert isinstance(task, PredictiveTask)
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


@pytest.mark.parametrize("batch_runtime", [EngineType.dspy], indirect=["batch_runtime"])
def test_serialization(summarization_docs, batch_runtime) -> None:
    pipe = Pipeline([
        summarization.Summarization(
            n_words=10,
            model=batch_runtime.model,
            generation_settings=batch_runtime.generation_settings,
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
                      'generation_settings': {'is_placeholder': False,
                                              'value': {
                                                        'config_kwargs': None,
                                                        'inference_kwargs': None,
                                                        'init_kwargs': None,
                                                        'strict_mode': False, 'inference_mode': None}},
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
    [EngineType.dspy, EngineType.langchain, EngineType.outlines],
    indirect=["batch_runtime"],
)
def test_inference_mode_override(batch_runtime) -> None:
    """Test that inference_mode parameter overrides the default value."""
    dummy = "dummy_inference_mode"

    task = summarization.Summarization(
        n_words=10,
        model=batch_runtime.model,
        generation_settings=GenerationSettings(inference_mode=dummy),
        batch_size=batch_runtime.batch_size,
    )

    assert task._bridge.inference_mode == dummy
