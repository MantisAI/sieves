from pathlib import Path

from sieves import Doc, Pipeline, tasks


def test_run() -> None:
    resources = [Doc(uri=Path(__file__).parent.parent.parent / "assets" / "1204.0162v2.pdf")]
    pipe = Pipeline(tasks=[tasks.preprocessing.Docling()])
    docs = list(pipe(resources))

    assert len(docs) == 1
    assert docs[0].text


def test_serialization() -> None:
    resources = [Doc(uri=Path(__file__).parent.parent.parent / "assets" / "1204.0162v2.pdf")]
    pipe = Pipeline(tasks=[tasks.preprocessing.Docling()])
    docs = list(pipe(resources))

    config = pipe.serialize()
    assert config.model_dump() == {
        "cls_name": "sieves.pipeline.core.Pipeline",
        "tasks": {
            "is_placeholder": False,
            "value": [
                {
                    "cls_name": "sieves.tasks.preprocessing.docling_.Docling",
                    "doc_converter": {"is_placeholder": True, "value": "docling.document_converter.DocumentConverter"},
                    "include_meta": {"is_placeholder": False, "value": False},
                    "show_progress": {"is_placeholder": False, "value": True},
                    "task_id": {"is_placeholder": False, "value": "Docling"},
                    "version": "0.7.0",
                }
            ],
        },
        "version": "0.7.0",
    }

    deserialized_pipeline = Pipeline.deserialize(config=config, tasks_kwargs=[{"doc_converter": None}])
    assert docs[0] == list(deserialized_pipeline(resources))[0]
