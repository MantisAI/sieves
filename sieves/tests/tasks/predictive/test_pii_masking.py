# mypy: ignore-errors
import pytest

from sieves import Doc, Pipeline, tasks
from sieves.model_wrappers import ModelType, ModelSettings, dspy_, langchain_, outlines_
from sieves.serialization import Config
from sieves.tasks import PredictiveTask, PIIMasking
from sieves.tasks.predictive import pii_masking


@pytest.mark.parametrize(
    "batch_runtime",
    PIIMasking.supports(),
    indirect=["batch_runtime"],
)
@pytest.mark.parametrize("fewshot", [True, False])
def test_run(pii_masking_docs, batch_runtime, fewshot) -> None:
    fewshot_examples = [
        pii_masking.FewshotExample(
            text="Jane Smith works at NASA.",
            masked_text="[MASKED] works at NASA.",
            pii_entities=[pii_masking.PIIEntity(entity_type="PERSON", text="Jane Smith")],
        ),
        pii_masking.FewshotExample(
            text="He lives at Diagon Alley 37.",
            masked_text="He lives at [MASKED].",
            pii_entities=[pii_masking.PIIEntity(entity_type="ADDRESS", text="Diagon Alley 37")],
        ),
    ]

    fewshot_args = {"fewshot_examples": fewshot_examples} if fewshot else {}
    task = tasks.predictive.PIIMasking(
        model=batch_runtime.model,
        model_settings=batch_runtime.model_settings,
        batch_size=batch_runtime.batch_size,
        **fewshot_args,
    )
    pipe = Pipeline(task)
    docs = list(pipe(pii_masking_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert "PIIMasking" in doc.results
        assert "PIIMasking" in doc.meta
        assert "raw" in doc.meta["PIIMasking"]
        assert "usage" in doc.meta["PIIMasking"]
        assert "usage" in doc.meta

        print(f"Output: {doc.results['PIIMasking']}")
        print(f"Raw output: {doc.meta['PIIMasking']['raw']}")
        print(f"Usage: {doc.meta['PIIMasking']['usage']}")
        print(f"Total Usage: {doc.meta['usage']}")

    with pytest.raises(NotImplementedError):
        pipe["PIIMasking"].distill(None, None, None, None, None, None, None, None)

    if fewshot:
        _to_hf_dataset(task, docs)


def _to_hf_dataset(task: PIIMasking, docs: list[Doc]) -> None:
    """Tests whether conversion to HF dataset works as expected.

    :param task: PIIMasking task instance.
    :param docs: List of documents to convert.
    """
    dataset = task.to_hf_dataset(docs)
    assert all([key in dataset.features for key in ("text", "masked_text")])
    assert len(dataset) == 2
    records = list(dataset)
    assert records[0]["text"] == "Her SSN is 222-333-444. Her credit card number is 1234 5678."
    assert records[1]["text"] == "You can reach Michael at michael.michaels@gmail.com."

    with pytest.raises(KeyError):
        task.to_hf_dataset([Doc(text="This is a dummy text.")])


@pytest.mark.parametrize("batch_runtime", [ModelType.dspy], indirect=["batch_runtime"])
def test_serialization(pii_masking_docs, batch_runtime) -> None:
    pipe = Pipeline([
        tasks.predictive.PIIMasking(
            model=batch_runtime.model,
            model_settings=batch_runtime.model_settings,
            batch_size=batch_runtime.batch_size,
        )
    ])

    config = pipe.serialize()
    assert config.model_dump() == {'cls_name': 'sieves.pipeline.core.Pipeline',
 'tasks': {'is_placeholder': False,
           'value': [{'cls_name': 'sieves.tasks.predictive.pii_masking.core.PIIMasking',
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
                      'mask_placeholder': {'is_placeholder': False,
                                           'value': '[MASKED]'},
                      'model': {'is_placeholder': True,
                                'value': 'dspy.clients.lm.LM'},
                      'pii_types': {'is_placeholder': False, 'value': None},
                      'prompt_instructions': {'is_placeholder': False,
                                          'value': None},
                      'task_id': {'is_placeholder': False,
                                  'value': 'PIIMasking'},
                      'condition': {'is_placeholder': False, 'value': None},
                      'version': Config.get_version()}]},
 'use_cache': {'is_placeholder': False, 'value': True},
 'version': Config.get_version()}

    Pipeline.deserialize(config=config, tasks_kwargs=[{"model": batch_runtime.model}])


@pytest.mark.parametrize(
    "batch_runtime",
    PIIMasking.supports(),
    indirect=["batch_runtime"],
)
def test_inference_mode_override(batch_runtime) -> None:
    """Test that inference_mode parameter overrides the default value."""
    dummy = "dummy_inference_mode"

    task = tasks.predictive.PIIMasking(
        model=batch_runtime.model,
        model_settings=ModelSettings(inference_mode=dummy),
        batch_size=batch_runtime.batch_size,
    )

    assert task._bridge.inference_mode == dummy


@pytest.mark.parametrize(
    "batch_runtime",
    PIIMasking.supports(),
    indirect=["batch_runtime"],
)
def test_run_with_dict_pii_types(pii_masking_docs, batch_runtime) -> None:
    """Test PIIMasking with dict format pii_types (labels with descriptions)."""
    pii_types_with_descriptions = {
        "EMAIL": "Email addresses",
        "PHONE": "Phone numbers",
        "SSN": "Social security numbers",
        "CREDIT_CARD": "Credit card numbers"
    }

    pipe = Pipeline([
        tasks.predictive.PIIMasking(
            pii_types=pii_types_with_descriptions,
            model=batch_runtime.model,
            model_settings=batch_runtime.model_settings,
            batch_size=batch_runtime.batch_size,
        )
    ])
    docs = list(pipe(pii_masking_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert "PIIMasking" in doc.results


@pytest.mark.parametrize(
    "batch_runtime",
    PIIMasking.supports(),
    indirect=["batch_runtime"],
)
def test_run_with_list_pii_types(pii_masking_docs, batch_runtime) -> None:
    """Test PIIMasking with list format pii_types (backward compatibility)."""
    pii_types_list = ["EMAIL", "PHONE", "SSN", "CREDIT_CARD"]

    pipe = Pipeline([
        tasks.predictive.PIIMasking(
            pii_types=pii_types_list,
            model=batch_runtime.model,
            model_settings=batch_runtime.model_settings,
            batch_size=batch_runtime.batch_size,
        )
    ])
    docs = list(pipe(pii_masking_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert "PIIMasking" in doc.results


@pytest.mark.parametrize(
    "batch_runtime",
    PIIMasking.supports(),
    indirect=["batch_runtime"],
)
def test_run_with_none_pii_types(pii_masking_docs, batch_runtime) -> None:
    """Test PIIMasking with None pii_types (auto-detect all PII types)."""
    pipe = Pipeline([
        tasks.predictive.PIIMasking(
            pii_types=None,
            model=batch_runtime.model,
            model_settings=batch_runtime.model_settings,
            batch_size=batch_runtime.batch_size,
        )
    ])
    docs = list(pipe(pii_masking_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert "PIIMasking" in doc.results
