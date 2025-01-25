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

    # Test docs-to-dataset conversion.
    task = pipe["classifier"]
    assert isinstance(task, PredictiveTask)
    dataset = task.docs_to_dataset(docs)
    assert all([key in dataset.features for key in ("text", "label")])
    assert len(dataset) == 2
    dataset_records = list(dataset)
    for rec in dataset_records:
        assert isinstance(rec["label"], list)
        for v in rec["label"]:
            assert isinstance(v, float)
        assert isinstance(rec["text"], str)

    with pytest.raises(KeyError):
        task.docs_to_dataset([Doc(text="This is a dummy text.")])
