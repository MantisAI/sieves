# mypy: ignore-errors
import pytest

from sieves import Pipeline
from sieves.engines import EngineType
from sieves.tasks.predictive import classification


@pytest.mark.parametrize(
    "engine,fewshot",
    [
        (EngineType.dspy, False),
        (EngineType.dspy, True),
        (EngineType.langchain, False),
        (EngineType.langchain, True),
        (EngineType.ollama, False),
        (EngineType.ollama, True),
        (EngineType.outlines, False),
        (EngineType.outlines, True),
    ],
    indirect=["engine"],
)
def test_fewshot_examples(dummy_docs, engine, fewshot):
    fewshot_examples = [
        classification.TaskFewshotExample(
            text="On the properties of hydrogen atoms and red dwarfs.",
            confidence_per_label={"science": 1.0, "politics": 0.0},
        ),
        classification.TaskFewshotExample(
            text="A parliament is elected by casting votes.", confidence_per_label={"science": 0, "politics": 1.0}
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
