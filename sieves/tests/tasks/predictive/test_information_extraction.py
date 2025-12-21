# mypy: ignore-errors
import gliner2
import pydantic
import pytest

from sieves import Doc, Pipeline, tasks
from sieves.model_wrappers import ModelType, ModelSettings, dspy_, langchain_, outlines_
from sieves.serialization import Config
from sieves.tasks import PredictiveTask, InformationExtraction
from sieves.tasks.predictive import information_extraction


class Person(pydantic.BaseModel, frozen=True):
    name: str
    age: pydantic.PositiveInt

class PersonNotFrozen(pydantic.BaseModel):
    name: str
    age: pydantic.PositiveInt

PersonGliner = gliner2.inference.engine.Schema().structure(
    "Person"
).field("name", dtype="str").field("age", dtype="str")

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
                entities=[Person(name="Ada Lovelace", age=47), Person(name="Zeno of Citium", age=72)],
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
                entity=Person(name="Ada Lovelace", age=47),
            ),
            information_extraction.FewshotExampleSingle(
                text="Alan Watts passed away at the age of 58 years.",
                entity=Person(name="Alan Watts", age=58),
            ),
        ]

    entity_type = PersonGliner if isinstance(batch_runtime.model, gliner2.GLiNER2) else Person
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

    # Ensure entity type checks work as expected.
    if entity_type is not PersonGliner:
        with pytest.raises(TypeError):
            tasks.predictive.InformationExtraction(
                entity_type=PersonGliner,
                model=batch_runtime.model,
                model_settings=batch_runtime.model_settings,
                batch_size=batch_runtime.batch_size,
                mode=mode,
                **fewshot_args
            )
    else:
        with pytest.raises(TypeError):
            tasks.predictive.InformationExtraction(
                entity_type=Person,
                model=batch_runtime.model,
                model_settings=batch_runtime.model_settings,
                batch_size=batch_runtime.batch_size,
                mode=mode,
                **fewshot_args
            )

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert "InformationExtraction" in doc.results
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
                doc.results["InformationExtraction"] is None or
                isinstance(doc.results["InformationExtraction"], pydantic.BaseModel)
            )
        else:
            assert isinstance(doc.results["InformationExtraction"], list)

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
        entity_type=PersonGliner if isinstance(batch_runtime.model, gliner2.GLiNER2) else Person,
        model=batch_runtime.model,
        model_settings=ModelSettings(inference_mode=dummy),
        batch_size=batch_runtime.batch_size,
    )

    assert task._bridge.inference_mode == dummy
