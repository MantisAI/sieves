# mypy: ignore-errors
import pytest

from sieves import Doc, Pipeline
from sieves.engines import EngineType
from sieves.tasks import PredictiveTask
from sieves.tasks.predictive import classification


@pytest.mark.parametrize("engine", EngineType.all(), indirect=["engine"])
@pytest.mark.parametrize("fewshot", [True, False])
def test_run(dummy_docs, engine, fewshot):
    fewshot_examples = [
        classification.TaskFewshotExample(
            text="On the properties of hydrogen atoms and red dwarfs.",
            reasoning="Atoms, hydrogen and red dwarfs are terms from physics. There is no mention of any "
            "politics-related terms.",
            confidence_per_label={"science": 1.0, "politics": 0.0},
        ),
        classification.TaskFewshotExample(
            text="A parliament is elected by casting votes.",
            reasoning="The election of a parliament by the casting of votes is a component of a democratic political "
            "system.",
            confidence_per_label={"science": 0, "politics": 1.0},
        ),
    ]

    fewshot_args = {"fewshot_examples": fewshot_examples} if fewshot else {}
    pipe = Pipeline(
        [
            classification.Classification(
                task_id="classifier", labels=["science", "politics"], engine=engine, **fewshot_args
            ),
        ]
    )
    docs = list(pipe(dummy_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert doc.results["classifier"]
        assert "classifier" in doc.results


@pytest.mark.parametrize("engine", [EngineType.huggingface], indirect=["engine"])
def test_to_dataset(dummy_docs, engine) -> None:
    task = classification.Classification(task_id="classifier", labels=["science", "politics"], engine=engine)

    assert isinstance(task, PredictiveTask)
    dataset = task.to_dataset(task(dummy_docs))
    assert all([key in dataset.features for key in ("text", "label")])
    assert len(dataset) == 2
    dataset_records = list(dataset)
    for rec in dataset_records:
        assert isinstance(rec["label"], list)
        for v in rec["label"]:
            assert isinstance(v, float)
        assert isinstance(rec["text"], str)

    with pytest.raises(KeyError):
        task.to_dataset([Doc(text="This is a dummy text.")])


@pytest.mark.parametrize("engine", [EngineType.huggingface], indirect=["engine"])
def test_serialization(dummy_docs, engine) -> None:
    pipe = Pipeline(
        [classification.Classification(task_id="classifier", labels=["science", "politics"], engine=engine)]
    )
    list(pipe(dummy_docs))

    config = pipe.serialize()
    assert config.model_dump() == {
        "cls_name": "sieves.pipeline.core.Pipeline",
        "tasks": {
            "is_placeholder": False,
            "value": [
                {
                    "cls_name": "sieves.tasks.predictive.classification.core.Classification",
                    "engine": {
                        "is_placeholder": False,
                        "value": {
                            "cls_name": "sieves.engines.huggingface_.HuggingFace",
                            "inference_kwargs": {"is_placeholder": False, "value": {}},
                            "init_kwargs": {"is_placeholder": False, "value": {}},
                            "model": {
                                "is_placeholder": True,
                                "value": "transformers.pipelines.zero_shot_classification."
                                "ZeroShotClassificationPipeline",
                            },
                            "version": "0.4.0",
                        },
                    },
                    "fewshot_examples": {"is_placeholder": False, "value": ()},
                    "include_meta": {"is_placeholder": False, "value": True},
                    "labels": {"is_placeholder": False, "value": ["science", "politics"]},
                    "prompt_signature_desc": {"is_placeholder": False, "value": None},
                    "prompt_template": {"is_placeholder": False, "value": None},
                    "show_progress": {"is_placeholder": False, "value": True},
                    "task_id": {"is_placeholder": False, "value": "classifier"},
                    "version": "0.4.0",
                }
            ],
        },
        "version": "0.4.0",
    }

    Pipeline.deserialize(config=config, tasks_kwargs=[{"engine": {"model": engine.model}}])
