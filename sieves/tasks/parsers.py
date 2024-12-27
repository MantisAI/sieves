"""File parsers for converting raw files into documents."""
from pathlib import Path
from typing import Any, Iterable, Optional

import docling.document_converter
from loguru import logger
from tqdm import tqdm

from sieves.data.doc import Doc
from sieves.data.resource import Resource
from sieves.tasks.core import PreTask


class DoclingParser(PreTask[Iterable[Resource]]):
    """Parser that uses docling to convert files into documents."""

    def __init__(
        self,
        doc_converter: docling.document_converter.DocumentConverter,
        task_id: Optional[str] = None,
        show_progress: bool = True,
    ):
        """Initialize the docling parser.
        :param doc_converter: Docling parser instance.
        :param show_progress: Whether to show progress bar for processed documents
        """
        super().__init__(task_id=task_id, show_progress=show_progress)
        self._doc_converter = doc_converter

    def __call__(self, resources: Iterable[Resource]) -> Iterable[Doc]:
        """Parse a set of files using docling.

        :param resources: Resources to process.
        :returns: Parsed documents
        """
        docs: list[Doc] = []
        resources = list(resources)

        # Wrap conversion in TQDM if progress should be shown.
        convert = self._doc_converter.convert_all
        if self._show_progress:

            def convert_with_progress(uris: Iterable[Path | str]) -> Any:
                return tqdm(self._doc_converter.convert_all(uris))

            convert = convert_with_progress

        parsed_resources = list(convert([resource.uri for resource in resources]))
        assert len(parsed_resources) == len(resources)

        for resource, parsed_resource in zip(resources, parsed_resources):
            try:
                doc = Doc(
                    content=parsed_resource,
                    chunks=None,
                    meta={
                        # Preserve existing metadata
                        **resource.meta,
                        self.id: {"docling_result": parsed_resource},
                    },
                )
                docs.append(doc)
            except Exception as e:
                logger.error(f"Failed to parse file {resource.uri}: {str(e)}")
                continue

        return docs
