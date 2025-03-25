# mypy: ignore-errors
import pytest
from PIL import Image

from sieves import Doc


@pytest.fixture  # type: ignore[misc]
def test_images() -> dict[str, Image.Image]:
    return {
        "rgb_red_100": Image.new("RGB", (100, 100), color="red"),
        "rgb_red_100_2": Image.new("RGB", (100, 100), color="red"),
        "rgb_blue_100": Image.new("RGB", (100, 100), color="blue"),
        "rgb_red_200": Image.new("RGB", (200, 200), color="red"),
        "l_gray_100": Image.new("L", (100, 100), color=128),
    }


@pytest.mark.image_comparison  # type: ignore[misc]
def test_identical_images(test_images: dict[str, Image.Image]) -> None:
    doc1 = Doc(images=[test_images["rgb_red_100"]])
    doc2 = Doc(images=[test_images["rgb_red_100_2"]])
    assert doc1 == doc2


@pytest.mark.image_comparison  # type: ignore[misc]
def test_different_images(test_images: dict[str, Image.Image]) -> None:
    doc1 = Doc(images=[test_images["rgb_red_100"]])
    doc2 = Doc(images=[test_images["rgb_blue_100"]])
    assert doc1 != doc2


@pytest.mark.image_comparison  # type: ignore[misc]
def test_none_images() -> None:
    doc1 = Doc(images=None)
    doc2 = Doc(images=None)
    assert doc1 == doc2


@pytest.mark.image_comparison  # type: ignore[misc]
def test_one_none_image(test_images: dict[str, Image.Image]) -> None:
    doc1 = Doc(images=[test_images["rgb_red_100"]])
    doc2 = Doc(images=None)
    assert doc1 != doc2


@pytest.mark.image_comparison  # type: ignore[misc]
def test_different_image_counts(test_images: dict[str, Image.Image]) -> None:
    doc1 = Doc(images=[test_images["rgb_red_100"], test_images["rgb_red_100_2"]])
    doc2 = Doc(images=[test_images["rgb_red_100"]])
    assert doc1 != doc2


@pytest.mark.image_comparison  # type: ignore[misc]
def test_different_image_sizes(test_images: dict[str, Image.Image]) -> None:
    doc1 = Doc(images=[test_images["rgb_red_100"]])
    doc2 = Doc(images=[test_images["rgb_red_200"]])
    assert doc1 != doc2


@pytest.mark.image_comparison  # type: ignore[misc]
def test_different_image_modes(test_images: dict[str, Image.Image]) -> None:
    doc1 = Doc(images=[test_images["rgb_red_100"]])
    doc2 = Doc(images=[test_images["l_gray_100"]])
    assert doc1 != doc2


@pytest.mark.image_comparison  # type: ignore[misc]
def test_doc_comparison_type_error() -> None:
    doc = Doc(images=None)
    with pytest.raises(NotImplementedError):
        doc == 42
