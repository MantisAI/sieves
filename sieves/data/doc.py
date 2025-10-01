"""Doc implementation, types and utilities."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Literal

from datasets import Dataset
from PIL import Image, ImageChops

Field = Literal["meta", "results", "uri", "text", "chunks", "id", "images"]


@dataclasses.dataclass
class Doc:
    """A document holding data to be processed."""

    meta: dict[str, Any] = dataclasses.field(default_factory=dict)
    results: dict[str, Any] = dataclasses.field(default_factory=dict)
    uri: Path | str | None = None
    text: str | None = None
    chunks: list[str] | None = None
    id: str | None = None
    images: list[Image.Image] | None = None

    def __post_init__(self) -> None:
        """Initialize chunks."""
        if self.chunks is None and self.text is not None:
            self.chunks = [self.text]

    @staticmethod
    def _are_images_equal(im1: Image.Image | None, im2: Image.Image | None) -> bool:
        """Check if two images are equal using PIL Image Channel operations.

        :param im1: First PIL image to compare.
        :param im2: Second PIL image to compare.
        :return bool: True if images are equal, False otherwise.
        """
        if im1 is None and im2 is None:
            return True
        if im1 is None or im2 is None:
            return False
        if im1.size != im2.size or im1.mode != im2.mode:
            return False
        return ImageChops.difference(im1, im2).getbbox() is None

    def __eq__(self, other: object) -> bool:
        """Compare two `Doc` instances.

        :return: True if `self` is equal to `other`.
        :raises NotImplementedError: if `other` isn't of type `Doc`.
        """
        if not isinstance(other, Doc):
            raise NotImplementedError

        # Check if images are equal
        images_equal_check = False
        if self.images is None and other.images is None:
            images_equal_check = True
        elif self.images is None or other.images is None:
            images_equal_check = False
        elif self.images is not None and other.images is not None:
            if len(self.images) == len(other.images):
                images_equal_check = all(
                    self._are_images_equal(im1, im2) for im1, im2 in zip(self.images, other.images)
                )
            else:
                images_equal_check = False
        return (
            self.id == other.id
            and self.uri == other.uri
            and self.text == other.text
            and self.chunks == other.chunks
            and self.results == other.results
            and images_equal_check
        )

    @classmethod
    def from_hf_dataset(cls, dataset: Dataset, column_map: dict[Field, Any] | None = None) -> list[Doc]:
        """Generate list of docs from Hugging Face `datasets.Dataset`.

        :param dataset: Dataset to generate `Doc` instances from. If column_map isn't specified to the contrary, dataset
            must contain at least one column named "text".
        :param column_map: Which `Doc` attribute to map to which attribute in `dataset`. If None, the mapping "text" ->
            "text" is assumed.
        :return: List of `Doc` instances, each representing one row in the dataset.
        :raises ValueError: If expected columns are not present in the dataset features.
        """
        if column_map is None:
            column_map = {"text": "text"}

        missing_cols = set(column_map.values()) - set(dataset.column_names)
        if len(missing_cols):
            raise KeyError(f"Specified columns '{missing_cols}' not found in dataset columns: {dataset.column_names}.")

        docs: list[Doc] = []
        for row in dataset:
            docs.append(cls(**{doc_col: row.get(data_col) for doc_col, data_col in column_map.items()}))  # type: ignore[misc]

        return docs
