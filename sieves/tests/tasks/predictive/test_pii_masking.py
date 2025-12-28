# mypy: ignore-errors
import pytest
from flaky import flaky

from sieves import Doc, Pipeline, tasks
from sieves.model_wrappers import ModelType, ModelSettings, dspy_, langchain_, outlines_
from sieves.serialization import Config
from sieves.tasks import PredictiveTask, PIIMasking
from sieves.tasks.predictive import pii_masking
from sieves.tasks.predictive.schemas.pii_masking import Result, PIIEntity


@pytest.mark.parametrize(
    "batch_runtime",
    PIIMasking.supports(),
    indirect=["batch_runtime"],
)
@pytest.mark.parametrize("fewshot", [True, False])
def test_run(pii_masking_docs, batch_runtime, fewshot) -> None:
    fewshot_examples = [
        pii_masking.FewshotExample(
            text="His name is John Doe and his SSN is 111-222-333.",
            masked_text="His name is [MASKED] and his SSN is [MASKED].",
            pii_entities=[
                pii_masking.PIIEntity(entity_type="PERSON", text="John Doe", score=1.0),
                pii_masking.PIIEntity(entity_type="SSN", text="111-222-333", score=1.0),
            ],
        ),
        pii_masking.FewshotExample(
            text="Contact Maria at maria.doe@gmail.com.",
            masked_text="Contact [MASKED] at [MASKED].",
            pii_entities=[
                pii_masking.PIIEntity(entity_type="PERSON", text="Maria", score=1.0),
                pii_masking.PIIEntity(entity_type="EMAIL", text="maria.doe@gmail.com", score=1.0),
            ],
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

        # Verify unified result types.
        res = doc.results["PIIMasking"]
        assert isinstance(res, pii_masking.Result)
        assert len(res.pii_entities)

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


@flaky(max_runs=3, min_passes=1)
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


@pytest.mark.parametrize("batch_runtime", [ModelType.outlines], indirect=["batch_runtime"])
def test_evaluation(batch_runtime) -> None:
    """Test evaluation for PII masking without running pipeline."""
    task = tasks.predictive.PIIMasking(model=batch_runtime.model, task_id="pii")

    # 1. Full overlap
    doc_full = Doc(text="My email is test@example.com")
    res_full = Result(
        masked_text="My email is [MASKED]",
        pii_entities=[PIIEntity(entity_type="EMAIL", text="test@example.com", start=12, end=28)]
    )
    doc_full.results["pii"] = res_full
    doc_full.gold["pii"] = res_full
    report_full = task.evaluate([doc_full])
    assert report_full.metrics[task.metric] == 1.0

    # 2. Partial overlap
    doc_partial = Doc(text="My email is test@example.com and phone is 123456")
    # Pred has both
    res_pred = Result(
        masked_text="My email is [MASKED] and phone is [MASKED]",
        pii_entities=[
            PIIEntity(entity_type="EMAIL", text="test@example.com", start=12, end=28),
            PIIEntity(entity_type="PHONE", text="123456", start=42, end=48)
        ]
    )
    # Gold has only email
    res_gold = Result(
        masked_text="My email is [MASKED] and phone is 123456",
        pii_entities=[PIIEntity(entity_type="EMAIL", text="test@example.com", start=12, end=28)]
    )
    doc_partial.results["pii"] = res_pred
    doc_partial.gold["pii"] = res_gold
    report_partial = task.evaluate([doc_partial])
    # TP=1, FP=1, FN=0 -> Precision=0.5, Recall=1.0 -> F1=0.666...
    assert 0.6 < report_partial.metrics[task.metric] < 0.7

    # 3. No overlap
    doc_none = Doc(text="My email is test@example.com")
    res_none_pred = Result(
        masked_text="My email is [MASKED]",
        pii_entities=[PIIEntity(entity_type="EMAIL", text="test@example.com", start=12, end=28)]
    )
    res_none_gold = Result(
        masked_text="My name is [MASKED]",
        pii_entities=[PIIEntity(entity_type="PERSON", text="John", start=11, end=15)]
    )
    doc_none.results["pii"] = res_none_pred
    doc_none.gold["pii"] = res_none_gold
    report_none = task.evaluate([doc_none])
    assert report_none.metrics[task.metric] == 0.0
