# mypy: ignore-errors
import pydantic
import pytest

from sieves import Doc, Pipeline
from sieves.engines import EngineType
from sieves.tasks.predictive import information_extraction


@pytest.mark.parametrize(
    "batch_engine", (EngineType.dspy, EngineType.langchain, EngineType.ollama), indirect=["batch_engine"]
)
@pytest.mark.parametrize("strict_mode", [True, False])
def test_strict_mode(batch_engine, strict_mode):
    batch_engine._strict_mode = strict_mode

    class Person(pydantic.BaseModel, frozen=True):
        name: str
        age: pydantic.PositiveInt

    pipe = Pipeline([information_extraction.InformationExtraction(entity_type=Person, engine=batch_engine)])

    docs: list[Doc] = []
    hit_exception = False
    if strict_mode:
        try:
            docs = list(pipe([Doc(text=".")]))
        except Exception:
            hit_exception = True
    if strict_mode is False:
        docs = list(pipe([Doc(text=".")]))

    if strict_mode and hit_exception:
        assert len(docs) == 0
    else:
        assert len(docs) == 1

    for doc in docs:
        assert "InformationExtraction" in doc.results
