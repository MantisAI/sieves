# mypy: ignore-errors
import pytest

from sieves import Pipeline
from sieves.engines import EngineType
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
