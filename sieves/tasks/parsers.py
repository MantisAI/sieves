"""File parsers for converting raw files into documents."""

from typing import Iterable

import docling.document_converter
from loguru import logger
from tqdm import tqdm

from sieves.data.doc import Doc
from sieves.data.resource import Resource
from sieves.tasks.core import PreTask


class DoclingParser(PreTask):
    """Parser that uses docling to convert files into documents."""

    def __init__(self, doc_converter: docling.document_converter.DocumentConverter, show_progress: bool = True):
        """Initialize the docling parser.
        :param doc_converter: Docling parser instance.
        :param show_progress: Whether to show progress bar for processed documents
        """

        super().__init__(show_progress=show_progress)
        self._id = "docling_parser"
        self._doc_converter = doc_converter

    @property
    def id(self) -> str:
        """Returns task ID.
        :returns: Task ID
        """
        return self._id

    def __call__(self, resources: Iterable[Resource], **kwargs) -> Iterable[Doc]:
        """Parse a set of files using docling.

        :param resources: Resources to process.
        :returns: Parsed documents
        """
        docs: list[Doc] = []
        resources = list(resources)

        # Wrap conversion in TQDM if progress should be shown.
        convert = self._doc_converter.convert_all
        if self._show_progress:
            convert = lambda uris: tqdm(self._doc_converter.convert_all(uris))  # noqa: E731

        parsed_resources = list(convert([resource.uri for resource in resources]))
        assert len(parsed_resources) == len(resources)

        for resource, parsed_resource in zip(resources, parsed_resources):
            try:
                doc = Doc(
                    content=parsed_resource.text,
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
