"""Marker task for converting PDF documents to text."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from tqdm import tqdm

from sieves.data import Doc
from sieves.tasks.core import Task


class Marker(Task):
    """Marker task for converting PDF documents to text with high accuracy."""

    def __init__(
        self,
        marker_converter: PdfConverter | None = None,
        task_id: str | None = None,
        show_progress: bool = True,
        include_meta: bool = False,
        extract_images: bool = False,
        config: dict[str, Any] | None = None,
    ):
        """Initialize the Marker task.

        :param marker_converter: Custom PdfConverter instance. If None, a default one will be created.
        :param task_id: Task ID.
        :param show_progress: Whether to show progress bar for processed documents.
        :param include_meta: Whether to include meta information generated by the task.
        :param extract_images: Whether to extract images from the PDF.
        :param config: Optional configuration for the PDF converter.
        """
        super().__init__(task_id=task_id, show_progress=show_progress, include_meta=include_meta)

        if marker_converter is None:
            # Use the simpler version if no custom converter is provided
            self._converter = PdfConverter(
                artifact_dict=create_model_dict(),
            )
        else:
            self._converter = marker_converter

        self._extract_images = extract_images
        self._config = config

    def __call__(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        """Process documents using Marker.

        :param docs: Documents to process.
        :return: Processed documents.
        """
        docs = list(docs)

        pbar: tqdm | None = tqdm(total=len(docs)) if self._show_progress else None
        try:
            for doc in docs:
                # Convert URI to string if it's a Path
                uri = str(doc.uri) if isinstance(doc.uri, Path) else doc.uri

                # Process the document
                rendered = self._converter(uri)

                # Extract text and optionally images
                if self._extract_images:
                    text, _, images = text_from_rendered(rendered)
                    # Store images in meta
                    doc.meta["images"] = images
                else:
                    text, _, _ = text_from_rendered(rendered)

                # Update document text
                doc.text = text
                if pbar:
                    pbar.update(1)
            return docs
        finally:
            if pbar:
                pbar.close()

    @property
    def _state(self) -> dict[str, Any]:
        """Get state for serialization.

        :return: State dictionary.
        """
        return {
            "extract_images": self._extract_images,
            "config": self._config,
        }
