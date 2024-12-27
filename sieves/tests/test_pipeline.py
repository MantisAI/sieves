import docling.document_converter

from sieves.tasks.parsers import DoclingParser


def test_pipeline() -> None:
    doc_converter = docling.document_converter.DocumentConverter()  # noqa: F841
    tasks = DoclingParser(doc_converter)  # noqa: F841
