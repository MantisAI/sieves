# mypy: ignore-errors
import pytest

from sieves import Pipeline, tasks
from sieves.engines import EngineType


@pytest.mark.parametrize(
    "engine",
    [EngineType.dspy, EngineType.glix, EngineType.huggingface, EngineType.ollama, EngineType.outlines],
    indirect=True,
)
def test_run(dummy_docs, engine) -> None:
    pipe = Pipeline(
        [
            # todo exchange with dummy task
            tasks.predictive.Classification(task_id="classifier", labels=["science", "politics"], engine=engine),
        ]
    )
    docs = list(pipe(dummy_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert doc.results["classifier"]
        assert "classifier" in doc.results
