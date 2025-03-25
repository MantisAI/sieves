from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from PIL import ImageChops
from PIL.Image import Image


@dataclasses.dataclass
class Doc:
    """A document holding data to be processed."""

    meta: dict[str, Any] = dataclasses.field(default_factory=dict)
    results: dict[str, Any] = dataclasses.field(default_factory=dict)
    uri: Path | str | None = None
    text: str | None = None
    chunks: list[str] | None = None
    id: str | None = None
    images: list[Image] | None = None

    def __post_init__(self) -> None:
        if self.chunks is None and self.text is not None:
            self.chunks = [self.text]

    @staticmethod
    def _are_images_equal(im1: Image, im2: Image) -> bool:
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
