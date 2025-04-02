# mypy: ignore-errors
import pytest

from sieves import Pipeline
from sieves.engines import EngineType
from sieves.serialization import Config
from sieves.tasks.predictive import ner
from sieves.tasks.predictive.ner.core import Entity


@pytest.mark.parametrize(
    "batch_engine",
    (
        EngineType.dspy,
        EngineType.instructor,
        EngineType.langchain,
        EngineType.ollama,
        EngineType.outlines,
        EngineType.glix,
    ),
    indirect=["batch_engine"],
)
@pytest.mark.parametrize("fewshot", [True, False])
def test_run(ner_docs, batch_engine, fewshot) -> None:
    fewshot_examples = [
        ner.TaskFewshotExample(
            text="John studied data science in Barcelona and lives with Jaume",
            entities=[
                Entity(text="John", context="John studied data", entity_type="PERSON"),
                Entity(text="Barcelona", context="science in Barcelona", entity_type="LOCATION"),
                Entity(text="Jaume", context="lives with Jaume", entity_type="PERSON"),
            ],
        ),
        ner.TaskFewshotExample(
            text="Maria studied computer engineering in Madrid and works with Carlos",
            entities=[
                Entity(text="Maria", context="Maria studied computer", entity_type="PERSON"),
                Entity(text="Madrid", context="engineering in Madrid and works", entity_type="LOCATION"),
                Entity(text="Carlos", context="works with Carlos", entity_type="PERSON"),
            ],
        ),
    ]

    fewshot_args = {"fewshot_examples": fewshot_examples} if fewshot else {}
    pipe = Pipeline(
        [
            ner.NER(entities=["PERSON", "LOCATION", "COMPANY"], engine=batch_engine, **fewshot_args),
        ]
    )
    docs = list(pipe(ner_docs))

    assert len(docs) == 2
    for doc in docs:
        assert "NER" in doc.results


@pytest.mark.parametrize("batch_engine", [EngineType.ollama], indirect=["batch_engine"])
def test_serialization(ner_docs, batch_engine) -> None:
    pipe = Pipeline([ner.NER(entities=["PERSON", "LOCATION", "COMPANY"], engine=batch_engine)])
    list(pipe(ner_docs))

    config = pipe.serialize()
    assert config.model_dump() == {
        "cls_name": "sieves.pipeline.core.Pipeline",
        "tasks": {
            "is_placeholder": False,
            "value": [
                {
                    "cls_name": "sieves.tasks.predictive.ner.core.NER",
                    "engine": {
                        "is_placeholder": False,
                        "value": {
                            "cls_name": "sieves.engines.wrapper.Engine",
                            "inference_kwargs": {"is_placeholder": False, "value": {}},
                            "init_kwargs": {"is_placeholder": False, "value": {}},
                            "model": {"is_placeholder": True, "value": "sieves.engines.ollama_.Model"},
                            "version": Config.get_version(),
                        },
                    },
                    "entities": {"is_placeholder": False, "value": ["PERSON", "LOCATION", "COMPANY"]},
                    "fewshot_examples": {"is_placeholder": False, "value": ()},
                    "include_meta": {"is_placeholder": False, "value": True},
                    "prompt_signature_desc": {"is_placeholder": False, "value": None},
                    "prompt_template": {"is_placeholder": False, "value": None},
                    "show_progress": {"is_placeholder": False, "value": True},
                    "task_id": {"is_placeholder": False, "value": "NER"},
                    "version": Config.get_version(),
                },
            ],
        },
        "version": Config.get_version(),
    }
    Pipeline.deserialize(
        config=config,
        tasks_kwargs=[{"engine": {"model": batch_engine.model}, "entities": ["PERSON", "LOCATION", "COMPANY"]}],
    )
