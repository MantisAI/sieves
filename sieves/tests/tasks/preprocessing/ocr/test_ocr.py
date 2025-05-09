# mypy: ignore-errors
from pathlib import Path

from docling.document_converter import DocumentConverter

from sieves import Doc, Pipeline, tasks
from sieves.serialization import Config


def test_run() -> None:
    resources = [Doc(uri=Path(__file__).parent.parent.parent.parent / "assets" / "1204.0162v2.pdf")]
    pipe = Pipeline(tasks=[tasks.preprocessing.OCR()])
    docs = list(pipe(resources))

    assert len(docs) == 1
    assert docs[0].text


def test_serialization() -> None:
    resources = [Doc(uri=Path(__file__).parent.parent.parent.parent / "assets" / "1204.0162v2.pdf")]
    pipe = Pipeline(tasks=[tasks.preprocessing.OCR()])
    config = pipe.serialize()
    version = Config.get_version()
    assert config.model_dump() == {
        "cls_name": "sieves.pipeline.core.Pipeline",
        "use_cache": {"is_placeholder": False, "value": True},
        "tasks": {
            "is_placeholder": False,
            "value": [
                {
                    "cls_name": "sieves.tasks.preprocessing.ocr.core.OCR",
                    "converter": {"is_placeholder": True, "value": "docling.document_converter.DocumentConverter"},
                    "export_format": {"is_placeholder": False, "value": "markdown"},
                    "show_progress": {"is_placeholder": False, "value": True},
                    "include_meta": {"is_placeholder": False, "value": False},
                    "task_id": {"is_placeholder": False, "value": "OCR"},
                    "version": version,
                }
            ],
        },
        "version": version,
    }

    # For deserialization, we need to provide the converter
    converter = DocumentConverter()
    deserialized_pipeline = Pipeline.deserialize(
        config=config, tasks_kwargs=[{"converter": converter, "export_format": "markdown"}]
    )
    deserialized_docs = list(deserialized_pipeline(resources))

    assert len(deserialized_docs) == 1
