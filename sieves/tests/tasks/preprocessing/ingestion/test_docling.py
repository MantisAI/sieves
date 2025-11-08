"""Tests for Docling task."""
from pathlib import Path

import pytest
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, smolvlm_picture_description
from docling.document_converter import DocumentConverter, ImageFormatOption
from docling_core.types.doc.document import DescriptionAnnotation

from sieves import Doc, Pipeline
from sieves.serialization import Config
from sieves.tasks import Ingestion
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
                    'batch_size': {'is_placeholder': False, "value": -1},
                    "converter": {"is_placeholder": True, "value": "docling.document_converter.DocumentConverter"},
                    "export_format": {"is_placeholder": False, "value": "markdown"},
                    "include_meta": {"is_placeholder": False, "value": False},
                    "task_id": {"is_placeholder": False, "value": "Docling"},
                    "version": version,
                    'condition': {'is_placeholder': False, 'value': None},
                }
            ],
        },
        "version": version,
    }

    deserialized_pipeline = Pipeline.deserialize(
        config=config, tasks_kwargs=[{"converter": None, "export_format": "markdown"}]
    )
    assert docs[0] == list(deserialized_pipeline(resources))[0]


@pytest.mark.slow
def test_image_description_with_local_vlm() -> None:
    """Test image description with local VLM model."""
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: ImageFormatOption(
                pipeline_options=PdfPipelineOptions(
                    do_picture_description=True,
                    picture_description_options=smolvlm_picture_description,
                    generate_picture_images=True,
                    images_scale=2.0
                )
            )
        }
    )

    resources = [Doc(uri=Path(__file__).parent.parent.parent.parent / "assets" / "1204.0162v2.pdf")]
    pipe = Pipeline(tasks=[Ingestion(converter=converter, include_meta=True)])
    docs = list(pipe(resources))

    assert len(docs) == 1
    assert docs[0].text
    pics_annotations = [pic.annotations for pic in docs[0].meta['Ingestion'].document.pictures]
    assert len(pics_annotations) == 6
    assert all(
        isinstance(pic_annotation, DescriptionAnnotation)
        for pic_annotations in pics_annotations for pic_annotation in pic_annotations
    )
    assert all(
        pic_annotation.text in docs[0].text
        for pic_annotations in pics_annotations for pic_annotation in pic_annotations
    )
