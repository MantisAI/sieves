# mypy: ignore-errors
import chonkie
import pytest

from sieves import Doc, Pipeline
from sieves.serialization import Config
from sieves.tasks.preprocessing import Chunking


@pytest.mark.parametrize("chunker", ["chonkie", "naive"])
def test_chonkie(chunker, tokenizer) -> None:
    resources = [Doc(text="This is a text. " * 100)]
    pipe = Pipeline(tasks=[Chunking(chonkie.TokenChunker(tokenizer) if chunker == "chonkie" else 5)])
    docs = list(pipe(resources))

    assert len(docs) == 1
    assert docs[0].text
    assert docs[0].chunks


def test_serialization(tokenizer) -> None:
    resources = [Doc(text="This is a text " * 100)]
    pipe = Pipeline(tasks=[Chunking(chonkie.TokenChunker(tokenizer))])
    docs = list(pipe(resources))

    config = pipe.serialize()
    assert config.model_dump() == {
        "cls_name": "sieves.pipeline.core.Pipeline",
        "use_cache": {"is_placeholder": False, "value": True},
        "tasks": {
            "is_placeholder": False,
            "value": [
                {
                    "chunker": {"is_placeholder": True, "value": "chonkie.chunker.token.TokenChunker"},
                    'batch_size': {'is_placeholder': False, "value": -1},
                    "cls_name": "sieves.tasks.preprocessing.chunking.core.Chunking",
                    "include_meta": {"is_placeholder": False, "value": False},
                    "task_id": {"is_placeholder": False, "value": "Chunking"},
                    "version": Config.get_version(),
                    'condition': {'is_placeholder': False, 'value': None},
                }
            ],
        },
        "version": Config.get_version(),
    }

    deserialized_pipeline = Pipeline.deserialize(
        config=config, tasks_kwargs=[{"chunker": chonkie.TokenChunker(tokenizer)}]
    )
    assert docs[0] == list(deserialized_pipeline(resources))[0]
