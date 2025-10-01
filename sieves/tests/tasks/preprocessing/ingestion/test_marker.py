# mypy: ignore-errors
from pathlib import Path

import pytest
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

from sieves import Doc, Pipeline, tasks
from sieves.serialization import Config


@pytest.mark.skip(reason="Currently running into OOM issues with instantiating Marker converts.")
def test_marker():
    """Workaround to keep memory usage low: run single function with one instantiated Marker instance."""
    marker_converter = PdfConverter(artifact_dict=create_model_dict())

    def test_run() -> None:
        resources = [Doc(uri=Path(__file__).parent.parent.parent / "assets" / "1204.0162v2.pdf")]
        pipe = Pipeline(tasks=[tasks.preprocessing.Marker(converter=marker_converter)])
        docs = list(pipe(resources))

        assert len(docs) == 1
        assert docs[0].text

    def test_with_extract_images() -> None:
        resources = [Doc(uri=Path(__file__).parent.parent.parent / "assets" / "1204.0162v2.pdf")]
        pipe = Pipeline(
            tasks=[tasks.preprocessing.Marker(converter=marker_converter, extract_images=True, include_meta=True)]
        )
        docs = list(pipe(resources))

        assert len(docs) == 1
        assert docs[0].text
        assert docs[0].images

    def test_serialization() -> None:
        resources = [Doc(uri=Path(__file__).parent.parent.parent / "assets" / "1204.0162v2.pdf")]
        pipe = Pipeline(tasks=[tasks.preprocessing.Marker(converter=marker_converter, include_meta=True)])
        docs = list(pipe(resources))

        config = pipe.serialize()
        version = Config.get_version()
        assert config.model_dump() == {
            "cls_name": "sieves.pipeline.core.Pipeline",
            "tasks": {
                "is_placeholder": False,
                "value": [
                    {
                        "cls_name": "sieves.tasks.preprocessing.marker_.Marker",
                        "converter": {"is_placeholder": True, "value": "marker.converters.pdf.PdfConverter"},
                        "export_format": {"is_placeholder": False, "value": "markdown"},
                        "extract_images": {"is_placeholder": False, "value": False},
                        "include_meta": {"is_placeholder": False, "value": True},
                        "task_id": {"is_placeholder": False, "value": "Marker"},
                        "version": version,
                    }
                ],
            },
            "version": version,
        }

        # For deserialization, we need to provide the converter
        converter = marker_converter
        deserialized_pipeline = Pipeline.deserialize(
            config=config, tasks_kwargs=[{"converter": converter, "export_format": "markdown"}]
        )
        deserialized_docs = list(deserialized_pipeline(resources))

        assert len(deserialized_docs) == 1
        assert deserialized_docs[0].text == docs[0].text

    test_run()
    test_with_extract_images()
    test_serialization()
