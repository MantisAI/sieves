# mypy: ignore-errors
import enum

import pytest

from sieves import Doc, Pipeline, engines
from sieves.engines import EngineType
from sieves.serialization import Config
from sieves.tasks import PredictiveTask
from sieves.tasks.predictive import classification


def _run(engine: engines.Engine, docs: list[Doc], fewshot: bool, multilabel: bool = True) -> None:
    assert issubclass(engine.inference_modes, enum.Enum)
    if multilabel:
        fewshot_examples = [
            classification.FewshotExampleMultiLabel(
                text="On the properties of hydrogen atoms and red dwarfs.",
                reasoning="Atoms, hydrogen and red dwarfs are terms from physics. There is no mention of any "
                "politics-related terms.",
                confidence_per_label={"science": 1.0, "politics": 0.0},
            ),
            classification.FewshotExampleMultiLabel(
                text="A parliament is elected by casting votes.",
                reasoning="The election of a parliament by the casting of votes is a component of a democratic "
                "political system.",
                confidence_per_label={"science": 0, "politics": 1.0},
            ),
        ]
    else:
        fewshot_examples = [
            classification.FewshotExampleSingleLabel(
                text="On the properties of hydrogen atoms and red dwarfs.",
                reasoning="Atoms, hydrogen and red dwarfs are terms from physics. There is no mention of any "
                "politics-related terms. This is about science - scientists, papers, experiments, laws of nature.",
                label="science",
                confidence=1.0,
            ),
            classification.FewshotExampleSingleLabel(
                text="A parliament is elected by casting votes.",
                reasoning="The election of a parliament by the casting of votes is a component of a democratic "
                "political system. This is about politics - parliament, laws, parties, politicians.",
                label="politics",
                confidence=1.0,
            ),
        ]

    fewshot_args = {"fewshot_examples": fewshot_examples} if fewshot else {}
    label_descriptions = {
        "science": "Topics related to scientific disciplines and research",
        "politics": "Topics related to government, elections, and political systems",
    }

    pipe = Pipeline(
        [
            classification.Classification(
                task_id="classifier",
                labels=["science", "politics"],
                engine=engine,
                label_descriptions=label_descriptions,
                multi_label=multilabel,
                **fewshot_args,
            ),
        ]
    )
    docs = list(pipe(docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert doc.results["classifier"]


@pytest.mark.parametrize("batch_engine", EngineType.all(), indirect=["batch_engine"])
@pytest.mark.parametrize("fewshot", [False])
@pytest.mark.parametrize("multilabel", [True])
def test_run(classification_docs, batch_engine, fewshot, multilabel):
    _run(batch_engine, classification_docs, fewshot, multilabel)


@pytest.mark.parametrize("engine", EngineType.all(), indirect=["engine"])
@pytest.mark.parametrize("fewshot", [True, False])
def test_run_nonbatched(classification_docs, engine, fewshot):
    _run(engine, classification_docs, fewshot)


@pytest.mark.parametrize("batch_engine", [EngineType.huggingface], indirect=["batch_engine"])
def test_to_hf_dataset(classification_docs, batch_engine) -> None:
    task = classification.Classification(task_id="classifier", labels=["science", "politics"], engine=batch_engine)

    assert isinstance(task, PredictiveTask)
    dataset = task.to_hf_dataset(task(classification_docs))
    assert all([key in dataset.features for key in ("text", "labels")])
    assert len(dataset) == 2
    dataset_records = list(dataset)
    for rec in dataset_records:
        assert isinstance(rec["labels"], list)
        for v in rec["labels"]:
            assert isinstance(v, int)
        assert isinstance(rec["text"], str)

    with pytest.raises(KeyError):
        task.to_hf_dataset([Doc(text="This is a dummy text.")])


@pytest.mark.parametrize("batch_engine", [EngineType.huggingface], indirect=["batch_engine"])
def test_serialization(classification_docs, batch_engine) -> None:
    label_descriptions = {
        "science": "Topics related to scientific disciplines and research",
        "politics": "Topics related to government, elections, and political systems",
    }

    pipe = Pipeline(
        classification.Classification(
            task_id="classifier",
            labels=["science", "politics"],
            engine=batch_engine,
            label_descriptions=label_descriptions,
        )
    )

    config = pipe.serialize()
    assert config.model_dump() == {
        "cls_name": "sieves.pipeline.core.Pipeline",
        "use_cache": {"is_placeholder": False, "value": True},
        "tasks": {
            "is_placeholder": False,
            "value": [
                {
                    "cls_name": "sieves.tasks.predictive.classification.core.Classification",
                    "engine": {
                        "is_placeholder": False,
                        "value": {
                            "batch_size": {"is_placeholder": False, "value": -1},
                            "cls_name": "sieves.engines.wrapper.Engine",
                            "inference_kwargs": {"is_placeholder": False, "value": {}},
                            "init_kwargs": {"is_placeholder": False, "value": {}},
                            "model": {
                                "is_placeholder": True,
                                "value": "transformers.pipelines.zero_shot_classification."
                                "ZeroShotClassificationPipeline",
                            },
                            "strict_mode": {"is_placeholder": False, "value": False},
                            "version": Config.get_version(),
                        },
                    },
                    "fewshot_examples": {"is_placeholder": False, "value": ()},
                    "include_meta": {"is_placeholder": False, "value": True},
                    "labels": {"is_placeholder": False, "value": ["science", "politics"]},
                    "prompt_signature_desc": {"is_placeholder": False, "value": None},
                    "prompt_template": {"is_placeholder": False, "value": None},
                    "show_progress": {"is_placeholder": False, "value": True},
                    "task_id": {"is_placeholder": False, "value": "classifier"},
                    "version": Config.get_version(),
                    "label_descriptions": {"is_placeholder": False, "value": label_descriptions},
                }
            ],
        },
        "version": Config.get_version(),
    }

    Pipeline.deserialize(config=config, tasks_kwargs=[{"engine": {"model": batch_engine.model}}])


@pytest.mark.parametrize("batch_engine", [EngineType.huggingface], indirect=["batch_engine"])
def test_label_descriptions_validation(batch_engine) -> None:
    """Test that invalid label descriptions raise a ValueError."""
    # Valid case - no label descriptions
    classification.Classification(
        labels=["science", "politics"],
        engine=batch_engine,
    )

    # Valid case - all labels have descriptions
    valid_descriptions = {"science": "Science related", "politics": "Politics related"}
    classification.Classification(
        labels=["science", "politics"], engine=batch_engine, label_descriptions=valid_descriptions
    )

    # Valid case - some labels have descriptions
    partial_descriptions = {"science": "Science related"}
    classification.Classification(
        labels=["science", "politics"], engine=batch_engine, label_descriptions=partial_descriptions
    )

    # Invalid case - description for non-existent label
    invalid_descriptions = {"science": "Science related", "economics": "Economics related"}
    with pytest.raises(ValueError, match="Label descriptions contain invalid labels"):
        classification.Classification(
            labels=["science", "politics"], engine=batch_engine, label_descriptions=invalid_descriptions
        )
