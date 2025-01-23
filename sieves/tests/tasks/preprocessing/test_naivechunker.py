# mypy: ignore-errors
import pytest

from sieves import Pipeline, tasks
from sieves.engines import EngineType


@pytest.mark.parametrize(
    "engine",
    [EngineType.huggingface],
    indirect=["engine"],
)
def test_task_chunking(dummy_docs, engine) -> None:
    """Tests whether chunking mechanism in PredictiveTask works as expected."""
    chunk_interval = 5
    pipe = Pipeline(
        [
            tasks.preprocessing.NaiveChunker(interval=chunk_interval),
            tasks.predictive.Classification(task_id="classifier", labels=["science", "politics"], engine=engine),
        ]
    )
    docs = list(pipe(dummy_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert doc.results["classifier"]
        assert len(doc.chunks) == 2
        assert "classifier" in doc.results
