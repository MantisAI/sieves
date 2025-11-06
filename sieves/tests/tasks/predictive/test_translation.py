# mypy: ignore-errors
import pytest

from sieves import Doc, Pipeline
from sieves.engines import EngineType, dspy_, langchain_, outlines_
from sieves.serialization import Config
from sieves.tasks import PredictiveTask
from sieves.tasks.predictive import translation


@pytest.mark.parametrize(
    "batch_runtime",
    (
        EngineType.dspy,
        EngineType.langchain,
        EngineType.outlines,
    ),
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
    pipe = Pipeline([
        translation.Translation(
            to="Spanish",
            model=batch_runtime.model,
            generation_settings=batch_runtime.generation_settings,
            batch_size=batch_runtime.batch_size,
            **fewshot_args,
        )
    ])
    docs = list(pipe(translation_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert "Translation" in doc.results

    with pytest.raises(NotImplementedError):
        pipe["Translation"].distill(None, None, None, None, None, None, None, None)


@pytest.mark.parametrize("batch_runtime", [EngineType.dspy], indirect=["batch_runtime"])
def test_to_hf_dataset(translation_docs, batch_runtime) -> None:
    task = translation.Translation(
        to="Spanish",
        model=batch_runtime.model,
        generation_settings=batch_runtime.generation_settings,
        batch_size=batch_runtime.batch_size,
    )
    pipe = Pipeline(task)
    docs = pipe(translation_docs)

    assert isinstance(task, PredictiveTask)
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


@pytest.mark.parametrize("batch_runtime", [EngineType.dspy], indirect=["batch_runtime"])
def test_serialization(translation_docs, batch_runtime) -> None:
    pipe = Pipeline([
        translation.Translation(
            to="Spanish",
            model=batch_runtime.model,
            generation_settings=batch_runtime.generation_settings,
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
                      'generation_settings': {'is_placeholder': False,
                                              'value': {
                                                        'config_kwargs': None,
                                                        'inference_kwargs': None,
                                                        'init_kwargs': None,
                                                        'strict_mode': False}},
                      'include_meta': {'is_placeholder': False, 'value': True},
                      'model': {'is_placeholder': True,
                                'value': 'dspy.clients.lm.LM'},
                      'prompt_instructions': {'is_placeholder': False,
                                          'value': None},
                      'task_id': {'is_placeholder': False,
                                  'value': 'Translation'},
                      'to': {'is_placeholder': False, 'value': 'Spanish'},
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
    # Select a non-default inference mode based on engine type
    if "dspy" in batch_runtime.model.__class__.__module__:
        custom_mode = dspy_.InferenceMode.chain_of_thought
    elif "langchain" in batch_runtime.model.__class__.__module__:
        custom_mode = langchain_.InferenceMode.structured
    elif "outlines" in batch_runtime.model.__class__.__module__:
        custom_mode = outlines_.InferenceMode.text
    else:
        raise ValueError(f"Unsupported engine type: {batch_runtime.model.__class__.__module__}")

    task = translation.Translation(
        to="Spanish",
        model=batch_runtime.model,
        generation_settings=batch_runtime.generation_settings,
        batch_size=batch_runtime.batch_size,
        inference_mode=custom_mode,
    )

    assert task._bridge.inference_mode == custom_mode
