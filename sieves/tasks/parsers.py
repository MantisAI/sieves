"""File parsers for converting raw files into documents."""

from typing import Iterable, Optional

import docling.document_converter
from loguru import logger
from tqdm import tqdm

from sieves.data.doc import Doc
from sieves.data.resource import Resource
from sieves.tasks.core import PreTask


class DoclingParser(PreTask):
    """Parser that uses docling to convert files into documents."""

    def __init__(
        self,
        doc_converter: docling.document_converter.DocumentConverter,
        task_id: Optional[str] = None,
        show_progress: bool = True,
    ):
        """Initialize Docling parser task.

        :param doc_converter: Docling DocConverter object.
        :param task_id: Task ID.
        :param show_progress: Whether to show progress bar for processed documents
        """
        super().__init__(task_id=task_id, show_progress=show_progress)
        self._doc_converter = doc_converter

    def __call__(self, resources: Iterable[Resource], **kwargs) -> Iterable[Doc]:
        """Parse a set of files using docling.
        :param resources: Files to process.
        :returns: Parsed documents.
        """
        # Example doc: "https://arxiv.org/pdf/2206.01062"
        docs: list[Doc] = []
        iterable = tqdm(resources) if self._progress_bar else resources

        for resource in iterable:
            try:
                docling_doc = self._doc_converter.convert(resource.uri)
                docs.append(
                    Doc(
                        content=docling_doc.text,
                        chunks=[],
                        meta={
                            # Preserve existing metadata
                            **resource.meta,
                            self.id: {"docling_doc": docling_doc},
                        },
                    )
                )
            except Exception as e:
                logger.error(f"Failed to parse resource {resource.uri}: {str(e)}")
                continue

        return docs
