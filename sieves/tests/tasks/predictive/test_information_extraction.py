# mypy: ignore-errors
import gliner2
import pydantic
import pytest
from flaky import flaky

from sieves import Doc, Pipeline, tasks
from sieves.model_wrappers import ModelType, ModelSettings, dspy_, langchain_, outlines_
from sieves.serialization import Config
from sieves.tasks import PredictiveTask, InformationExtraction
from sieves.tasks.predictive import information_extraction
from sieves.tasks.predictive.schemas.information_extraction import ResultSingle, ResultMulti


class Person(pydantic.BaseModel, frozen=True):
    name: str
    age: pydantic.PositiveInt
    score: pydantic.NonNegativeFloat | None = None

class PersonNotFrozen(pydantic.BaseModel):
    name: str
    age: pydantic.PositiveInt
    score: pydantic.NonNegativeFloat | None = None

@flaky(max_runs=3, min_passes=1)
@pytest.mark.parametrize(
    "batch_runtime",
    InformationExtraction.supports(),
    indirect=["batch_runtime"],
)
@pytest.mark.parametrize("fewshot", [True, False])
@pytest.mark.parametrize("mode", ["multi", "single"])
def test_run(information_extraction_docs, batch_runtime, fewshot, mode) -> None:
    if mode == "multi":
        fewshot_examples = [
            information_extraction.FewshotExampleMulti(
                text="Ada Lovelace lived to 47 years old. Zeno of Citium died with 72 years.",
                entities=[Person(name="Ada Lovelace", age=47, score=1.), Person(name="Zeno of Citium", age=72)],
            ),
            information_extraction.FewshotExampleMulti(
                text="Alan Watts passed away at the age of 58 years. Alan Watts was 58 years old at the time of his death.",
                entities=[Person(name="Alan Watts", age=58)],
            ),
    ]

    else:
        fewshot_examples = [
            information_extraction.FewshotExampleSingle(
                text="Ada Lovelace lived to 47 years old.",
                entity=Person(name="Ada Lovelace", age=47, score=1.),
            ),
            information_extraction.FewshotExampleSingle(
                text="Alan Watts passed away at the age of 58 years.",
                entity=Person(name="Alan Watts", age=58),
            ),
        ]

    entity_type = Person
    fewshot_args = {"fewshot_examples": fewshot_examples} if fewshot else {}
    task = tasks.predictive.InformationExtraction(
        entity_type=entity_type,
        model=batch_runtime.model,
        model_settings=batch_runtime.model_settings,
        batch_size=batch_runtime.batch_size,
        mode=mode,
        **fewshot_args
    )

    pipe = Pipeline(task)
    docs = list(pipe(information_extraction_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert "InformationExtraction" in doc.results

        # Verify unified result types.
        res = doc.results["InformationExtraction"]
        if mode == "multi":
            assert isinstance(res, information_extraction.ResultMulti)
            assert len(res.entities)
        else:
            assert isinstance(res, information_extraction.ResultSingle)
            assert res.entity is not None

        assert "InformationExtraction" in doc.meta
        assert "raw" in doc.meta["InformationExtraction"]
        assert "usage" in doc.meta["InformationExtraction"]
        assert "usage" in doc.meta

        print(f"Output: {doc.results['InformationExtraction']}")
        print(f"Raw output: {doc.meta['InformationExtraction']['raw']}")
        print(f"Usage: {doc.meta['InformationExtraction']['usage']}")
        print(f"Total Usage: {doc.meta['usage']}")

        if mode == "single":
            assert (
                doc.results["InformationExtraction"].entity is None or
                isinstance(doc.results["InformationExtraction"].entity, pydantic.BaseModel)
            )
        else:
            assert isinstance(doc.results["InformationExtraction"].entities, list)

    with pytest.raises(NotImplementedError):
        pipe["InformationExtraction"].distill(None, None, None, None, None, None, None, None)

    if fewshot_examples:
        _to_hf_dataset(task, docs, mode)


def _to_hf_dataset(task: InformationExtraction, docs: list[Doc], mode: str) -> None:
    """Tests whether conversion to HF dataset works as expected.

    :param task: InformationExtraction task instance.
    :param docs: List of documents to convert.
    :param mode: Extraction mode.
    """
    assert isinstance(task, PredictiveTask)
    dataset = task.to_hf_dataset(docs)
    target_field = "entities" if mode == "multi" else "entity"
    assert all([key in dataset.features for key in ("text", target_field)])
    assert len(dataset) == 2
    records = list(dataset)
    assert records[0]["text"] == "Mahatma Ghandi lived to 79 years old. Bugs Bunny is at least 85 years old."
    assert records[1]["text"] == "Marie Curie passed away at the age of 67 years. Marie Curie was 67 years old."
    for record in records:
        res = record[target_field]
        if mode == "multi":
            assert isinstance(res, dict)
            assert isinstance(res["age"], list)
            assert isinstance(res["name"], list)
        else:
            assert res is None or isinstance(res, dict)

    with pytest.raises(KeyError):
        task.to_hf_dataset([Doc(text="This is a dummy text.")])

@pytest.mark.parametrize(
    "batch_runtime",
    [ModelType.outlines],
    indirect=["batch_runtime"],
)
def test_frozen_check(batch_runtime) -> None:
    # This should work (frozen)
    tasks.predictive.InformationExtraction(
        entity_type=Person,
        model=batch_runtime.model,
    )

    # This should raise ValueError (not frozen)
    with pytest.raises(ValueError, match="isn't frozen"):
        tasks.predictive.InformationExtraction(
            entity_type=PersonNotFrozen,
            model=batch_runtime.model,
        )


@pytest.mark.parametrize("batch_runtime", [ModelType.outlines], indirect=["batch_runtime"])
@pytest.mark.parametrize("mode", ["multi", "single"])
def test_serialization(information_extraction_docs, batch_runtime, mode) -> None:
    pipe = Pipeline(
        tasks.predictive.InformationExtraction(
            entity_type=Person,
            model=batch_runtime.model,
            model_settings=batch_runtime.model_settings,
            batch_size=batch_runtime.batch_size,
            mode=mode,
        )
    )

    config = pipe.serialize()
    assert config.model_dump() == {
        'cls_name': 'sieves.pipeline.core.Pipeline',
        'tasks': {
            'is_placeholder': False,
            'value': [{
                'cls_name': 'sieves.tasks.predictive.information_extraction.core.InformationExtraction',
                'entity_type': {'is_placeholder': True, 'value': 'pydantic._internal._model_construction.ModelMetaclass'},
                'fewshot_examples': {'is_placeholder': False, 'value': ()},
                'batch_size': {'is_placeholder': False, "value": -1},
                'mode': {'is_placeholder': False, 'value': mode},
                'model_settings': {
                    'is_placeholder': False,
                    'value': {
                        'config_kwargs': None,
                        'inference_kwargs': None,
                        'init_kwargs': None,
                        'strict': True,
                        'inference_mode': None
                    }
                },
                'include_meta': {'is_placeholder': False, 'value': True},
                'model': {'is_placeholder': True, 'value': 'outlines.models.openai.OpenAI'},
                'prompt_instructions': {'is_placeholder': False, 'value': None},
                'task_id': {'is_placeholder': False, 'value': 'InformationExtraction'},
                'condition': {'is_placeholder': False, 'value': None},
                'version': Config.get_version()
            }]
        },
        'use_cache': {'is_placeholder': False, 'value': True},
        'version': Config.get_version()
    }

    Pipeline.deserialize(config=config, tasks_kwargs=[{"model": batch_runtime.model, "entity_type": Person, "mode": mode}])


@pytest.mark.parametrize(
    "batch_runtime",
    InformationExtraction.supports(),
    indirect=["batch_runtime"],
)
def test_inference_mode_override(batch_runtime) -> None:
    """Test that inference_mode parameter overrides the default value."""
    dummy = "dummy_inference_mode"

    task = tasks.predictive.InformationExtraction(
        entity_type=Person,
        model=batch_runtime.model,
        model_settings=ModelSettings(inference_mode=dummy),
        batch_size=batch_runtime.batch_size,
    )

    assert task._bridge.inference_mode == dummy


@pytest.mark.parametrize("batch_runtime", [ModelType.outlines], indirect=["batch_runtime"])
def test_evaluation(batch_runtime) -> None:
    """Test evaluation for information extraction without running pipeline."""
    task = tasks.predictive.InformationExtraction(entity_type=Person, model=batch_runtime.model, task_id="ie", mode="single")

    # 1. Full overlap (Single)
    doc_full = Doc(text="Ada is 47.")
    res_full = ResultSingle(entity=Person(name="Ada", age=47))
    doc_full.results["ie"] = res_full
    doc_full.gold["ie"] = res_full
    report_full = task.evaluate([doc_full])
    assert report_full.metrics[task.metric] == 1.0

    # 2. No overlap (Single)
    doc_none = Doc(text="Ada is 47.")
    res_none_pred = ResultSingle(entity=Person(name="Alan", age=58))
    doc_none.results["ie"] = res_none_pred
    doc_none.gold["ie"] = res_full
    report_none = task.evaluate([doc_none])
    assert report_none.metrics[task.metric] == 0.0

    # 3. Multi-label partial overlap
    task_multi = tasks.predictive.InformationExtraction(
        entity_type=Person, model=batch_runtime.model, task_id="ie_multi", mode="multi"
    )
    doc_partial = Doc(text="Ada is 47 and Alan is 58.")
    # Pred has both
    res_multi_pred = ResultMulti(entities=[Person(name="Ada", age=47), Person(name="Alan", age=58)])
    # Gold has only Ada
    res_multi_gold = ResultMulti(entities=[Person(name="Ada", age=47)])
    doc_partial.results["ie_multi"] = res_multi_pred
    doc_partial.gold["ie_multi"] = res_multi_gold
    report_partial = task_multi.evaluate([doc_partial])
    # TP=1, FP=1, FN=0 -> Precision=0.5, Recall=1.0 -> F1=0.666...
    assert 0.6 < report_partial.metrics[task_multi.metric] < 0.7
