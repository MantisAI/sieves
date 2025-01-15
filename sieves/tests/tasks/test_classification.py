# mypy: ignore-errors
import pytest

from sieves import Pipeline
from sieves.engines import EngineType
from sieves.tasks.predictive import classification


@pytest.mark.parametrize(
    "engine",
    [EngineType.dspy, EngineType.glix, EngineType.huggingface, EngineType.ollama, EngineType.outlines],
    indirect=True,
)
def test_fewshot_examples(dummy_docs, engine):
    fewshot_examples = [
        classification.TaskFewshotExample(
            text="On the properties of hydrogen atoms and red dwarfs.",
            confidence_per_label={"science": 1.0, "politics": 0.0},
        ),
        classification.TaskFewshotExample(
            text="A parliament is elected by casting votes.", confidence_per_label={"science": 0, "politics": 1.0}
        ),
    ]
    pipe = Pipeline(
        [
            classification.Classification(
                task_id="classifier", labels=["science", "politics"], engine=engine, fewshot_examples=fewshot_examples
            ),
        ]
    )
    docs = list(pipe(dummy_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert doc.results["classifier"]
        assert "classifier" in doc.results
