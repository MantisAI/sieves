# mypy: ignore-errors
import pytest

from sieves import Doc, Pipeline
from sieves.model_wrappers import ModelType, ModelSettings, dspy_, langchain_, outlines_
from sieves.serialization import Config
from sieves.tasks import PredictiveTask
from sieves.tasks.predictive import translation


@pytest.mark.parametrize(
    "batch_runtime",
    translation.Translation.supports(),
    indirect=["batch_runtime"],
)
@pytest.mark.parametrize("fewshot", [True, False])
def test_run(translation_docs, batch_runtime, fewshot) -> None:
    fewshot_examples = [
        translation.FewshotExample(
            text="The sun is shining today.",
            to="Spanish",
            translation="El sol brilla hoy.",
        ),
        translation.FewshotExample(
            text="There's a lot of fog today",
            to="Spanish",
            translation="Hay mucha niebla hoy.",
        ),
    ]

    fewshot_args = {"fewshot_examples": fewshot_examples} if fewshot else {}
    task = translation.Translation(
        to="Spanish",
        model=batch_runtime.model,
        model_settings=batch_runtime.model_settings,
        batch_size=batch_runtime.batch_size,
        **fewshot_args,
    )
    pipe = Pipeline(task)
    docs = list(pipe(translation_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert "Translation" in doc.results
        assert "Translation" in doc.meta
        assert "raw" in doc.meta["Translation"]
        assert "usage" in doc.meta["Translation"]
        assert "usage" in doc.meta

        print(f"Output: {doc.results['Translation']}")
        print(f"Raw output: {doc.meta['Translation']['raw']}")
        print(f"Usage: {doc.meta['Translation']['usage']}")
        print(f"Total Usage: {doc.meta['usage']}")

    with pytest.raises(NotImplementedError):
        pipe["Translation"].distill(None, None, None, None, None, None, None, None)

    if fewshot:
        _to_hf_dataset(task, docs)


def _to_hf_dataset(task: translation.Translation, docs: list[Doc]) -> None:
    """Tests whether conversion to HF dataset works as expected.

    :param task: Translation task instance.
    :param docs: List of documents to convert.
    """
    dataset = task.to_hf_dataset(docs)
    assert all([key in dataset.features for key in ("text", "translation")])
    assert len(dataset) == 2
    records = list(dataset)
    assert records[0]["text"] == "It is rainy today."
    assert records[1]["text"] == "It is cloudy today."
    for record in records:
        assert isinstance(record["translation"], str)

    with pytest.raises(KeyError):
        task.to_hf_dataset([Doc(text="This is a dummy text.")])


@pytest.mark.parametrize("batch_runtime", [ModelType.dspy], indirect=["batch_runtime"])
def test_serialization(translation_docs, batch_runtime) -> None:
    pipe = Pipeline([
        translation.Translation(
            to="Spanish",
            model=batch_runtime.model,
            model_settings=batch_runtime.model_settings,
            batch_size=batch_runtime.batch_size,
        )
    ])

    config = pipe.serialize()
    assert config.model_dump() == {'cls_name': 'sieves.pipeline.core.Pipeline',
 'tasks': {'is_placeholder': False,
           'value': [{'cls_name': 'sieves.tasks.predictive.translation.core.Translation',
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
                                  'value': 'Translation'},
                      'condition': {'is_placeholder': False, 'value': None},
                      'to': {'is_placeholder': False, 'value': 'Spanish'},
                      'version': Config.get_version()}]},
 'use_cache': {'is_placeholder': False, 'value': True},
 'version': Config.get_version()}

    Pipeline.deserialize(config=config, tasks_kwargs=[{"model": batch_runtime.model}])


@pytest.mark.parametrize(
    "batch_runtime",
    translation.Translation.supports(),
    indirect=["batch_runtime"],
)
def test_inference_mode_override(batch_runtime) -> None:
    """Test that inference_mode parameter overrides the default value."""
    dummy = "dummy_inference_mode"

    task = translation.Translation(
        to="Spanish",
        model=batch_runtime.model,
        model_settings=ModelSettings(inference_mode=dummy),
        batch_size=batch_runtime.batch_size,
    )

    assert task._bridge.inference_mode == dummy
