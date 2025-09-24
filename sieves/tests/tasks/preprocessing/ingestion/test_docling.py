from pathlib import Path

from sieves import Doc, Pipeline
from sieves.serialization import Config
from sieves.tasks.preprocessing.ingestion.docling_ import Docling


def test_run() -> None:
    resources = [Doc(uri=Path(__file__).parent.parent.parent.parent / "assets" / "1204.0162v2.pdf")]
    pipe = Pipeline(tasks=[Docling()])
    docs = list(pipe(resources))

    assert len(docs) == 1
    assert docs[0].text


def test_serialization() -> None:
    resources = [Doc(uri=Path(__file__).parent.parent.parent.parent / "assets" / "1204.0162v2.pdf")]
    pipe = Pipeline(tasks=[Docling()])
    docs = list(pipe(resources))

    config = pipe.serialize()
    version = Config.get_version()
    assert config.model_dump() == {
        "cls_name": "sieves.pipeline.core.Pipeline",
        "use_cache": {"is_placeholder": False, "value": True},
        "tasks": {
            "is_placeholder": False,
            "value": [
                {
                    "cls_name": "sieves.tasks.preprocessing.ingestion.docling_.Docling",
                    "converter": {"is_placeholder": True, "value": "docling.document_converter.DocumentConverter"},
                    "export_format": {"is_placeholder": False, "value": "markdown"},
                    "include_meta": {"is_placeholder": False, "value": False},
                    "task_id": {"is_placeholder": False, "value": "Docling"},
                    "version": version,
                }
            ],
        },
        "version": version,
    }

    deserialized_pipeline = Pipeline.deserialize(
        config=config, tasks_kwargs=[{"converter": None, "export_format": "markdown"}]
    )
    assert docs[0] == list(deserialized_pipeline(resources))[0]
