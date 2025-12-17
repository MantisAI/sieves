# mypy: ignore-errors
import pytest

from sieves import Pipeline
from sieves.model_wrappers import ModelType
from sieves.serialization import Config
from sieves.tasks.preprocessing.chunking.naive import NaiveChunker


@pytest.mark.parametrize(
    "batch_runtime",
    [ModelType.huggingface],
    indirect=["batch_runtime"],
)
def test_run(dummy_docs, batch_runtime) -> None:
    """Tests whether chunking mechanism in PredictiveTask works as expected."""
    chunk_interval = 5
    pipe = Pipeline([NaiveChunker(interval=chunk_interval)])
    docs = list(pipe(dummy_docs))

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert len(doc.chunks) == 2


def test_serialization(dummy_docs) -> None:
    chunk_interval = 5
    pipe = Pipeline(tasks=[NaiveChunker(interval=chunk_interval)])
    docs = list(pipe(dummy_docs))

    config = pipe.serialize()
    assert config.model_dump() == {
        "cls_name": "sieves.pipeline.core.Pipeline",
        "use_cache": {"is_placeholder": False, "value": True},
        "tasks": {
            "is_placeholder": False,
            "value": [
                {
                    "cls_name": "sieves.tasks.preprocessing.chunking.naive.NaiveChunker",
                    'batch_size': {'is_placeholder': False, "value": -1},
                    "include_meta": {"is_placeholder": False, "value": False},
                    "interval": {"is_placeholder": False, "value": 5},
                    "task_id": {"is_placeholder": False, "value": "NaiveChunker"},
                    "version": Config.get_version(),
                    'condition': {'is_placeholder': False, 'value': None},
                }
            ],
        },
        "version": Config.get_version(),
    }

    deserialized_pipeline = Pipeline.deserialize(config=config, tasks_kwargs=[{}])
    assert docs[0] == list(deserialized_pipeline(dummy_docs))[0]
