# mypy: ignore-errors
import pytest
from flaky import flaky

from sieves import Doc, Pipeline
from sieves.model_wrappers import ModelType, ModelSettings
from sieves.serialization import Config
from sieves.tasks.predictive import ner
from sieves.tasks.predictive.schemas.ner import EntityWithContext, TaskResult, Entity


@flaky(max_runs=3, min_passes=1)
@pytest.mark.parametrize(
    "batch_runtime",
    ner.NER.supports(),
    indirect=["batch_runtime"],
)
@pytest.mark.parametrize("fewshot", [True, False])
def test_run(ner_docs, batch_runtime, fewshot) -> None:
    fewshot_examples = [
        ner.FewshotExample(
            text="John studied data science in Barcelona and lives with Jaume",
            entities=[
                EntityWithContext(text="John", context="John studied data", entity_type="PERSON", score=1.0),
                EntityWithContext(text="Barcelona", context="science in Barcelona", entity_type="LOCATION", score=1.0),
                EntityWithContext(text="Jaume", context="lives with Jaume", entity_type="PERSON", score=1.0),
            ],
        ),
        ner.FewshotExample(
            text="Maria studied computer engineering in Madrid and works with Carlos",
            entities=[
                EntityWithContext(text="Maria", context="Maria studied computer", entity_type="PERSON", score=1.0),
                EntityWithContext(text="Madrid", context="engineering in Madrid and works", entity_type="LOCATION", score=1.0),
                EntityWithContext(text="Carlos", context="works with Carlos", entity_type="PERSON", score=1.0),
            ],
        ),
    ]

    fewshot_args = {"fewshot_examples": fewshot_examples} if fewshot else {}
    task = ner.NER(
        entities=["PERSON", "LOCATION", "COMPANY"],
        model=batch_runtime.model,
        model_settings=batch_runtime.model_settings,
        batch_size=batch_runtime.batch_size,
        **fewshot_args
    )
    pipe = Pipeline(task)
    docs = list(pipe(ner_docs))

    assert len(docs) == 2
    for doc in docs:
        assert "NER" in doc.results

        # Verify unified result types.
        assert isinstance(doc.results["NER"], ner.Result)

        assert "NER" in doc.meta
        assert "raw" in doc.meta["NER"]
        assert "usage" in doc.meta["NER"]
        assert "usage" in doc.meta

        print(f"Output: {doc.results['NER']}")
        print(f"Raw output: {doc.meta['NER']['raw']}")
        print(f"Usage: {doc.meta['NER']['usage']}")
        print(f"Total Usage: {doc.meta['usage']}")

    with pytest.raises(NotImplementedError):
        pipe["NER"].distill(None, None, None, None, None, None, None, None)

    if fewshot:
        _to_hf_dataset(task, docs)


@pytest.mark.parametrize("batch_runtime", [ModelType.dspy], indirect=["batch_runtime"])
def test_serialization(ner_docs, batch_runtime) -> None:
    pipe = Pipeline(
        ner.NER(
            entities=["PERSON", "LOCATION", "COMPANY"],
            model=batch_runtime.model,
            model_settings=batch_runtime.model_settings,
            batch_size=batch_runtime.batch_size,
        )
    )

    config = pipe.serialize()
    assert config.model_dump() == {'cls_name': 'sieves.pipeline.core.Pipeline',
 'tasks': {'is_placeholder': False,
           'value': [{'cls_name': 'sieves.tasks.predictive.ner.core.NER',
                      'entities': {'is_placeholder': False,
                                   'value': ['PERSON', 'LOCATION', 'COMPANY']},
                      'fewshot_examples': {'is_placeholder': False,
                                           'value': ()},
                      'batch_size': {'is_placeholder': False, "value": -1},
                      'model_settings': {'is_placeholder': False,
                                              'value': {
                                                        'config_kwargs': None,
                                                        'inference_kwargs': None,
                                                        'init_kwargs': None,
                                                        'strict': True,
                                                        'inference_mode': None,}},
                      'include_meta': {'is_placeholder': False, 'value': True},
                      'model': {'is_placeholder': True,
                                'value': 'dspy.clients.lm.LM'},
                      'prompt_instructions': {'is_placeholder': False,
                                          'value': None},
                      'task_id': {'is_placeholder': False, 'value': 'NER'},
                      'condition': {'is_placeholder': False, 'value': None},
                      'version': Config.get_version()}]},
 'use_cache': {'is_placeholder': False, 'value': True},
 'version': Config.get_version()}
    Pipeline.deserialize(
        config=config,
        tasks_kwargs=[{"model": batch_runtime.model}],
    )


def _to_hf_dataset(task: ner.NER, docs: list[Doc]) -> None:
    """Tests whether conversion to HF dataset works as expected.

    :param task: NER task instance.
    :param docs: List of documents to convert.
    """
    dataset = task.to_hf_dataset(docs)
    assert all([key in dataset.features for key in ("text", "entities")])
    assert len(dataset) == 2
    dataset_records = list(dataset)
    for rec in dataset_records:
        assert isinstance(rec["entities"], dict)
        assert (
            len(rec["entities"]["entity_type"])
            == len(rec["entities"]["start"])
            == len(rec["entities"]["end"])
            == len(rec["entities"]["text"])
        )
        assert isinstance(rec["text"], str)

    with pytest.raises(KeyError):
        task.to_hf_dataset([Doc(text="This is a dummy text.")])


@pytest.mark.parametrize(
    "batch_runtime",
    [ModelType.dspy, ModelType.langchain, ModelType.outlines, ModelType.gliner],
    indirect=["batch_runtime"],
)
def test_inference_mode_override(batch_runtime) -> None:
    """Test that inference_mode parameter overrides the default value."""
    dummy = "dummy_inference_mode"

    task = ner.NER(
        entities=["PERSON", "LOCATION", "COMPANY"],
        model=batch_runtime.model,
        model_settings=ModelSettings(inference_mode=dummy),
        batch_size=batch_runtime.batch_size,
    )

    assert task._bridge.inference_mode == dummy


@pytest.mark.parametrize(
    "batch_runtime",
    ner.NER.supports(),
    indirect=["batch_runtime"],
)
def test_run_with_dict_entities(ner_docs, batch_runtime) -> None:
    """Test NER with dict format entities (labels with descriptions)."""
    entities_with_descriptions = {
        "PERSON": "Names of people",
        "LOCATION": "Geographic locations",
        "COMPANY": "Names of companies and organizations"
    }

    pipe = Pipeline(
        ner.NER(
            entities=entities_with_descriptions,
            model=batch_runtime.model,
            model_settings=batch_runtime.model_settings,
            batch_size=batch_runtime.batch_size,
        )
    )
    docs = list(pipe(ner_docs))

    assert len(docs) == 2
    for doc in docs:
        assert "NER" in doc.results


@pytest.mark.parametrize("batch_runtime", [ModelType.gliner], indirect=["batch_runtime"])
def test_evaluation(batch_runtime) -> None:
    """Test evaluation for NER without running pipeline."""
    task = ner.NER(entities=["PERSON", "LOCATION"], model=batch_runtime.model, task_id="ner")

    # 1. Full overlap
    doc_full = Doc(text="John lives in Berlin.")
    res_full = TaskResult(
        text=doc_full.text,
        entities=[
            Entity(text="John", start=0, end=4, entity_type="PERSON"),
            Entity(text="Berlin", start=14, end=20, entity_type="LOCATION")
        ]
    )
    doc_full.results["ner"] = res_full
    doc_full.gold["ner"] = res_full
    report_full = task.evaluate([doc_full])
    assert report_full.metrics[task.metric] == 1.0

    # 2. Partial overlap
    doc_partial = Doc(text="John lives in Berlin.")
    # Pred has John and Berlin
    res_pred = TaskResult(
        text=doc_partial.text,
        entities=[
            Entity(text="John", start=0, end=4, entity_type="PERSON"),
            Entity(text="Berlin", start=14, end=20, entity_type="LOCATION")
        ]
    )
    # Gold has John and Paris (Berlin is missing, Paris is extra in gold)
    res_gold = TaskResult(
        text=doc_partial.text,
        entities=[
            Entity(text="John", start=0, end=4, entity_type="PERSON"),
            Entity(text="Paris", start=14, end=19, entity_type="LOCATION")
        ]
    )
    doc_partial.results["ner"] = res_pred
    doc_partial.gold["ner"] = res_gold
    report_partial = task.evaluate([doc_partial])
    # Precision: 1 TP (John) / 2 Pred = 0.5
    # Recall: 1 TP (John) / 2 Gold = 0.5
    # F1: 0.5. (Note: _evaluate_dspy_example uses the logic in NER subclass which computes F1)
    assert report_partial.metrics[task.metric] == 0.5

    # 3. No overlap
    doc_none = Doc(text="John lives in Berlin.")
    res_none_pred = TaskResult(
        text=doc_none.text, entities=[Entity(text="John", start=0, end=4, entity_type="PERSON")]
    )
    res_none_gold = TaskResult(
        text=doc_none.text, entities=[Entity(text="Berlin", start=14, end=20, entity_type="LOCATION")]
    )
    doc_none.results["ner"] = res_none_pred
    doc_none.gold["ner"] = res_none_gold
    report_none = task.evaluate([doc_none])
    assert report_none.metrics[task.metric] == 0.0
