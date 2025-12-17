# mypy: ignore-errors
import pydantic
import pytest

from sieves import Doc, Pipeline, ModelSettings
from sieves.model_wrappers import ModelType
from sieves.tasks.predictive import information_extraction


@pytest.mark.parametrize(
    "batch_runtime", (ModelType.dspy, ModelType.langchain, ModelType.outlines), indirect=["batch_runtime"]
)
@pytest.mark.parametrize("strict", [True, False])
def test_strict(batch_runtime, strict):
    model_settings = ModelSettings(strict=strict)

    class Person(pydantic.BaseModel, frozen=True):
        name: str
        age: pydantic.PositiveInt

    pipe = Pipeline([
        information_extraction.InformationExtraction(
            entity_type=Person,
            model=batch_runtime.model,
            model_settings=model_settings,
            batch_size=batch_runtime.batch_size,
        )
    ])

    docs: list[Doc] = []
    hit_exception = False
    if strict:
        try:
            docs = list(pipe([Doc(text=".")]))
        except Exception:
            hit_exception = True
    if strict is False:
        docs = list(pipe([Doc(text=".")]))

    if strict and hit_exception:
        assert len(docs) == 0
    else:
        assert len(docs) == 1

    for doc in docs:
        assert "InformationExtraction" in doc.results
