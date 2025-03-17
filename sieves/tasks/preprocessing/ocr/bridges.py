from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import docling.datamodel.document
import docling.document_converter
import nltk
import unstructured
import unstructured.documents.elements
import unstructured.partition.auto
from loguru import logger

from sieves.data.doc import Doc

# Type definitions for unstructured
PartitionType = Callable[..., list[unstructured.documents.elements.Text]]
CleanerType = Callable[[str], str]


@dataclass
class OCRResult:
    """
    Dataclass to store OCR processing results.
    """

    text: str
    chunks: list[str] | None = None
    metadata: dict[str, Any] | None = None


class OCRBridge(abc.ABC):
    """
    Base bridge class for OCR engines.
    This class defines the interface that all OCR implementations must follow.
    """

    def __init__(self, task_id: str, **kwargs: Any):
        """
        Initialize the OCR bridge.
        :param task_id: Task ID.
        :param kwargs: Additional arguments specific to the OCR implementation.
        """
        self._task_id = task_id
        self._kwargs = kwargs

    @abc.abstractmethod
    def process(self, doc: Doc) -> OCRResult:
        """
        Process a document to extract text using OCR.
        :param doc: Document to process.
        :return: OCRResult containing extracted text, optional chunks, and metadata.
        """


class UnstructuredBridge(OCRBridge):
    """
    Bridge implementation for the Unstructured library.
    """

    def __init__(
        self,
        task_id: str,
        partition: PartitionType = unstructured.partition.auto.partition,
        cleaners: tuple[CleanerType, ...] = (),
        **kwargs: Any,
    ):
        """
        Initialize the Unstructured bridge.
        :param task_id: Task ID.
        :param partition: Function to use for partitioning.
        :param cleaners: Cleaning functions to apply.
        :param kwargs: Additional arguments to pass to the partition function.
        """
        super().__init__(task_id=task_id)
        self._partition = partition
        self._partition_args = kwargs or {}
        self._cleaners = cleaners

        # Download necessary resources
        UnstructuredBridge._require()

    @staticmethod
    def _require() -> None:
        """Download all necessary resources that have to be installed from within Python."""
        # Some nltk resources seem necessary for basic functionality.
        for nltk_resource in ("punkt", "averaged_perceptron_tagger"):
            # Don't install if already available.
            try:
                nltk.data.find(nltk_resource)
            except LookupError:
                nltk.download(nltk_resource)

    def process(self, doc: Doc) -> OCRResult:
        """
        Process a document using Unstructured.
        :param doc: Document to process.
        :return: OCRResult with extracted text.
        """
        if not doc.uri:
            raise ValueError("Documents have to have a value for .uri.")

        try:
            # Parse document
            parsed_resources: list[unstructured.documents.elements.Text] = self._partition(
                doc.uri, **self._partition_args
            )

            # Apply specified cleaners
            for cleaner in self._cleaners:
                for pr in parsed_resources:
                    pr.apply(cleaner)

            # Extract text and chunks
            text = "\n".join(resource.text for resource in parsed_resources)
            does_chunking = "chunking_strategy" in self._partition_args
            chunks = [pr.text for pr in parsed_resources] if does_chunking else None

            # Create metadata
            metadata = {
                "num_elements": len(parsed_resources),
                "element_types": [type(pr).__name__ for pr in parsed_resources],
                "parsed_resources": parsed_resources,
            }

            return OCRResult(text=text, chunks=chunks, metadata=metadata)

        except FileNotFoundError as err:
            raise FileNotFoundError(
                f"File at {doc.uri} not found. Ensure that this is a local file path - unstructured doesn't support"
                f" loading files via network URIs."
            ) from err


class DoclingBridge(OCRBridge):
    """
    Bridge implementation for the Docling library.
    Based on the Docling class from sieves/tasks/preprocessing/docling_.py.
    """

    def __init__(
        self,
        task_id: str,
        doc_converter: docling.document_converter.DocumentConverter | None = None,
    ):
        """
        Initialize the Docling bridge.
        :param task_id: Task ID.
        :param doc_converter: Docling document converter instance.
        """
        super().__init__(task_id=task_id)
        self._doc_converter = doc_converter if doc_converter else docling.document_converter.DocumentConverter()

    def process(self, doc: Doc) -> OCRResult:
        """
        Process a document using Docling.
        :param doc: Document to process.
        :return: OCRResult with extracted text.
        """
        if not doc.uri:
            raise ValueError("Documents have to have a value for .uri.")

        try:
            # Parse document
            parsed_resource: docling.datamodel.document.ConversionResult = self._doc_converter.convert(doc.uri)

            # Extract markdown text
            text = parsed_resource.document.export_to_markdown()

            # Extract metadata
            metadata: dict[str, Any] = {}

            return OCRResult(text=text, metadata=metadata)

        except Exception as e:
            logger.error(f"Failed to parse file {doc.uri}: {str(e)}")
            raise RuntimeError(f"Failed to parse file {doc.uri}: {str(e)}") from e
