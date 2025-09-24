# mypy: ignore-errors
import pydantic
import pytest

from sieves import Doc, Pipeline, tasks
from sieves.engines import EngineType
from sieves.serialization import Config
from sieves.tasks import PredictiveTask
from sieves.tasks.predictive import information_extraction


class Person(pydantic.BaseModel, frozen=True):
    name: str
    age: pydantic.PositiveInt


@pytest.mark.parametrize(
    "batch_engine",
    (
        EngineType.dspy,
        EngineType.instructor,
        EngineType.langchain,
        EngineType.ollama,
        EngineType.outlines,
        # EngineType.vllm,
    ),
    indirect=["batch_engine"],
)
@pytest.mark.parametrize("fewshot", [True, False])
def test_run(information_extraction_docs, batch_engine, fewshot) -> None:
    fewshot_examples = [
        information_extraction.FewshotExample(
            text="Ada Lovelace lived to 47 years old. Zeno of Citium died with 72 years.",
            reasoning="There is mention of two people in this text, including lifespans. I will extract those.",
            entities=[Person(name="Ada Loveloace", age=47), Person(name="Zeno of Citium", age=72)],
        ),
        information_extraction.FewshotExample(
            text="Alan Watts passed away at the age of 58 years. Alan Watts was 58 years old at the time of his death.",
            reasoning="There is mention of one person in this text, including lifespan. I will extract this person.",
            entities=[Person(name="Alan Watts", age=58)],
        ),
    ]

    fewshot_args = {"fewshot_examples": fewshot_examples} if fewshot else {}
    pipe = Pipeline(
        [
            tasks.predictive.InformationExtraction(entity_type=Person, engine=batch_engine, **fewshot_args),
        ]
    )
    docs = list(pipe(information_extraction_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert "InformationExtraction" in doc.results

    with pytest.raises(NotImplementedError):
        pipe["InformationExtraction"].distill(None, None, None, None, None, None, None, None)


@pytest.mark.parametrize("batch_engine", [EngineType.ollama], indirect=["batch_engine"])
def test_to_hf_dataset(information_extraction_docs, batch_engine) -> None:
    task = tasks.predictive.InformationExtraction(entity_type=Person, engine=batch_engine)
    docs = task(information_extraction_docs)

    assert isinstance(task, PredictiveTask)
    dataset = task.to_hf_dataset(docs)
    assert all([key in dataset.features for key in ("text", "entities")])
    assert len(dataset) == 2
    records = list(dataset)
    assert records[0]["text"] == "Mahatma Ghandi lived to 79 years old. Bugs Bunny is at least 85 years old."
    assert records[1]["text"] == "Marie Curie passed away at the age of 67 years. Marie Curie was 67 years old."
    for record in records:
        assert isinstance(record["entities"], dict)
        assert isinstance(record["entities"]["age"], list)
        assert isinstance(record["entities"]["name"], list)

    with pytest.raises(KeyError):
        task.to_hf_dataset([Doc(text="This is a dummy text.")])


@pytest.mark.parametrize("batch_engine", [EngineType.ollama], indirect=["batch_engine"])
def test_serialization(information_extraction_docs, batch_engine) -> None:
    pipe = Pipeline([tasks.predictive.InformationExtraction(entity_type=Person, engine=batch_engine)])

    config = pipe.serialize()
    assert config.model_dump() == {
        "cls_name": "sieves.pipeline.core.Pipeline",
        "use_cache": {"is_placeholder": False, "value": True},
        "tasks": {
            "is_placeholder": False,
            "value": [
                {
                    "cls_name": "sieves.tasks.predictive.information_extraction.core.InformationExtraction",
                    "engine": {
                        "is_placeholder": False,
                        "value": {
                            "cls_name": "sieves.engines.wrapper.Engine",
                            "inference_kwargs": {"is_placeholder": False, "value": {}},
                            "init_kwargs": {"is_placeholder": False, "value": {}},
                            "model": {"is_placeholder": True, "value": "sieves.engines.ollama_.Model"},
                            "batch_size": {"is_placeholder": False, "value": -1},
                            "strict_mode": {"is_placeholder": False, "value": False},
                            "version": Config.get_version(),
                        },
                    },
                    "entity_type": {
                        "is_placeholder": True,
                        "value": "pydantic._internal._model_construction.ModelMetaclass",
                    },
                    "fewshot_examples": {"is_placeholder": False, "value": ()},
                    "include_meta": {"is_placeholder": False, "value": True},
                    "prompt_signature_desc": {"is_placeholder": False, "value": None},
                    "prompt_template": {"is_placeholder": False, "value": None},
                    "task_id": {"is_placeholder": False, "value": "InformationExtraction"},
                    "version": Config.get_version(),
                }
            ],
        },
        "version": Config.get_version(),
    }

    Pipeline.deserialize(config=config, tasks_kwargs=[{"engine": {"model": batch_engine.model}, "entity_type": Person}])
