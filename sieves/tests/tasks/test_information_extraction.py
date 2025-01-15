# mypy: ignore-errors
import pydantic
import pytest

from sieves import Pipeline, tasks
from sieves.engines import EngineType
from sieves.tasks.predictive import information_extraction


@pytest.mark.parametrize(
    "engine,fewshot",
    [
        (EngineType.dspy, False),
        (EngineType.dspy, True),
        (EngineType.ollama, False),
        (EngineType.ollama, True),
        (EngineType.outlines, False),
        (EngineType.outlines, True),
    ],
    indirect=["engine"],
)
def test_run(information_extraction_docs, engine, fewshot) -> None:
    class Person(pydantic.BaseModel, frozen=True):
        name: str
        age: pydantic.PositiveInt

    fewshot_examples = [
        information_extraction.TaskFewshotExample(
            text="Mahatma Ghandi lived to 79 years old. Bugs Bunny is at least 85 years old.",
            entities=[Person(name="Mahatma Ghandi", age=79), Person(name="Bugs Bunny", age=85)],
        ),
        information_extraction.TaskFewshotExample(
            text="Marie Curie passed away with 67 years. Marie Curie was 67 years old.",
            entities=[
                Person(name="Marie Curie", age=67),
            ],
        ),
    ]

    fewshot_args = {"fewshot_examples": fewshot_examples} if fewshot else {}
    pipe = Pipeline(
        [
            tasks.predictive.InformationExtraction(entity_type=Person, engine=engine, **fewshot_args),
        ]
    )
    docs = list(pipe(information_extraction_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert "InformationExtraction" in doc.results
