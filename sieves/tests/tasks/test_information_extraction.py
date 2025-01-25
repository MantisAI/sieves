# mypy: ignore-errors
import pydantic
import pytest

from sieves import Doc, Pipeline, tasks
from sieves.engines import EngineType
from sieves.tasks import PredictiveTask
from sieves.tasks.predictive import information_extraction


@pytest.mark.parametrize(
    "engine", (EngineType.dspy, EngineType.langchain, EngineType.ollama, EngineType.outlines), indirect=["engine"]
)
@pytest.mark.parametrize("fewshot", [True, False])
def test_run(information_extraction_docs, engine, fewshot) -> None:
    class Person(pydantic.BaseModel, frozen=True):
        name: str
        age: pydantic.PositiveInt

    fewshot_examples = [
        information_extraction.TaskFewshotExample(
            text="Ada Lovelace lived to 47 years old. Zeno of Citium died with 72 years.",
            reasoning="There is mention of two people in this text, including lifespans. I will extract those.",
            entities=[Person(name="Ada Loveloace", age=47), Person(name="Zeno of Citium", age=72)],
        ),
        information_extraction.TaskFewshotExample(
            text="Alan Watts passed away with 58 years. Alan Watts was 58 years old at the time of his death.",
            reasoning="There is mention of one person in this text, including lifespan. I will extract this person.",
            entities=[Person(name="Alan Watts", age=58)],
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

    # Test docs-to-dataset conversion.
    task = pipe["InformationExtraction"]
    assert isinstance(task, PredictiveTask)
    dataset = task.docs_to_dataset(docs)
    assert all([key in dataset.features for key in ("text", "entities")])
    assert len(dataset) == 2
    records = list(dataset)
    assert records[0]["text"] == "Mahatma Ghandi lived to 79 years old. Bugs Bunny is at least 85 years old."
    assert records[1]["text"] == "Marie Curie passed away with 67 years. Marie Curie was 67 years old."
    for record in records:
        assert isinstance(record["entities"], dict)
        assert isinstance(record["entities"]["age"], list)
        assert isinstance(record["entities"]["name"], list)

    with pytest.raises(KeyError):
        task.docs_to_dataset([Doc(text="This is a dummy text.")])
