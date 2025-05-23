# mypy: ignore-errors
import chonkie

from sieves import Doc, Pipeline
from sieves.serialization import Config
from sieves.tasks.preprocessing.chunking import Chonkie


def test_chonkie(tokenizer) -> None:
    resources = [Doc(text="This is a text " * 100)]
    pipe = Pipeline(tasks=[Chonkie(chonkie.TokenChunker(tokenizer))])
    docs = list(pipe(resources))

    assert len(docs) == 1
    assert docs[0].text
    assert docs[0].chunks


def test_serialization(tokenizer) -> None:
    resources = [Doc(text="This is a text " * 100)]
    pipe = Pipeline(tasks=[Chonkie(chonkie.TokenChunker(tokenizer))])
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
                    "cls_name": "sieves.tasks.preprocessing.chunking.chonkie_.Chonkie",
                    "include_meta": {"is_placeholder": False, "value": False},
                    "show_progress": {"is_placeholder": False, "value": True},
                    "task_id": {"is_placeholder": False, "value": "Chonkie"},
                    "version": Config.get_version(),
                }
            ],
        },
        "version": Config.get_version(),
    }

    deserialized_pipeline = Pipeline.deserialize(
        config=config, tasks_kwargs=[{"chunker": chonkie.TokenChunker(tokenizer)}]
    )
    assert docs[0] == list(deserialized_pipeline(resources))[0]
