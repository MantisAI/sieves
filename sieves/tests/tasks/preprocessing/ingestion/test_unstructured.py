# mypy: ignore-errors
from pathlib import Path

import pytest
import unstructured.cleaners.core
import unstructured.partition.auto

from sieves import Doc, Pipeline
from sieves.serialization import Config
from sieves.tasks.preprocessing.ingestion.unstructured_ import Unstructured


@pytest.mark.parametrize("to_chunk", [True, False])
def test_run(to_chunk) -> None:
    resources = [Doc(uri=Path(__file__).parent.parent.parent.parent / "assets" / "1204.0162v2.pdf")]
    partition_kwargs = {"chunking_strategy": "basic"} if to_chunk else {}
    pipe = Pipeline(
        tasks=[
            Unstructured(
                **partition_kwargs,
                cleaners=(
                    lambda t: unstructured.cleaners.core.clean(
                        t, extra_whitespace=True, dashes=True, bullets=True, trailing_punctuation=True
                    ),
                ),
                include_meta=True,
            ),
        ]
    )
    docs = list(pipe(resources))

    assert len(docs) == 1
    assert docs[0].text
    if to_chunk:
        assert len(docs[0].chunks)
    else:
        assert docs[0].chunks is None


def test_serialization() -> None:
    resources = [Doc(uri=Path(__file__).parent.parent.parent.parent / "assets" / "dummy.txt")]

    def cleaner(text: str) -> str:
        return unstructured.cleaners.core.clean(
            text, extra_whitespace=True, dashes=True, bullets=True, trailing_punctuation=True
        )

    pipe = Pipeline(tasks=[Unstructured(cleaners=(cleaner,), include_meta=True)])
    docs = list(pipe(resources))

    config = pipe.serialize()
    assert config.model_dump() == {
        "cls_name": "sieves.pipeline.core.Pipeline",
        "use_cache": {"is_placeholder": False, "value": True},
        "tasks": {
            "is_placeholder": False,
            "value": [
                {
                    "cleaners": {"is_placeholder": True, "value": "builtins.tuple"},
                    "cls_name": "sieves.tasks.preprocessing.ingestion.unstructured_.Unstructured",
                    "include_meta": {"is_placeholder": False, "value": True},
                    "partition": {"is_placeholder": True, "value": "builtins.function"},
                    "task_id": {"is_placeholder": False, "value": "Unstructured"},
                    "version": Config.get_version(),
                }
            ],
        },
        "version": Config.get_version(),
    }

    deserialized_pipeline = Pipeline.deserialize(
        config=config, tasks_kwargs=({"partition": unstructured.partition.auto.partition, "cleaners": (cleaner,)},)
    )
    assert docs[0] == list(deserialized_pipeline(resources))[0]
