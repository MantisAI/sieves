"""File parsers for converting raw files into documents."""
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import docling.datamodel.document
import docling.document_converter
from loguru import logger
from tqdm import tqdm

from sieves.data.doc import Doc
from sieves.serialization import Attribute
from sieves.tasks.core import Task


class Docling(Task):
    """Parser wrappign the docling library to convert files into documents."""

    def __init__(
        self,
        doc_converter: docling.document_converter.DocumentConverter | None = None,
        task_id: str | None = None,
        show_progress: bool = True,
        include_meta: bool = False,
    ):
        """Initialize the docling parser.
        :param doc_converter: Docling parser instance.
        :param task_id: Task ID.
        :param show_progress: Whether to show progress bar for processed documents
        :param include_meta: Whether to include meta information generated by the task.
        """
        super().__init__(task_id=task_id, show_progress=show_progress, include_meta=include_meta)
        self._doc_converter = doc_converter if doc_converter else docling.document_converter.DocumentConverter()

    def __call__(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        """Parse resources using docling.

        :param docs: Resources to process.
        :returns: Parsed documents
        """
        docs = list(docs)

        # Validate docs.
        have_text = False
        for doc in docs:
            assert doc.uri, ValueError("Documents have to have a value for .uri.")
            if doc.text:
                have_text = True
        if have_text:
            warnings.warn(f"Task {self._task_id} is about to overwrite existing .text values.")

        # Wrap conversion in TQDM if progress should be shown.
        convert = self._doc_converter.convert_all
        if self._show_progress:

            def convert_with_progress(uris: Iterable[Path | str]) -> Any:
                return tqdm(self._doc_converter.convert_all(uris), total=len(docs))

            convert = convert_with_progress

        parsed_resources: list[docling.datamodel.document.ConversionResult] = list(
            convert([resource.uri for resource in docs])
        )
        assert len(parsed_resources) == len(docs)

        for doc, parsed_resource in zip(docs, parsed_resources):
            try:
                if self._include_meta:
                    doc.meta |= {self.id: parsed_resource}
                doc.text = parsed_resource.document.export_to_markdown()
            except Exception as e:
                logger.error(f"Failed to parse file {doc.uri}: {str(e)}")
                continue

        return docs

    @property
    def _attributes(self) -> dict[str, Attribute]:
        return {
            **super()._attributes,
            "doc_converter": Attribute(value=self._doc_converter, is_placeholder=True),
        }
