# mypy: ignore-errors
import pytest
import regex
from datasets import Dataset
from PIL import Image

from sieves import Doc


@pytest.fixture
def test_images() -> dict[str, Image.Image]:
    return {
        "rgb_red_100": Image.new("RGB", (100, 100), color="red"),
        "rgb_red_100_2": Image.new("RGB", (100, 100), color="red"),
        "rgb_blue_100": Image.new("RGB", (100, 100), color="blue"),
        "rgb_red_200": Image.new("RGB", (200, 200), color="red"),
        "l_gray_100": Image.new("L", (100, 100), color=128),
    }


def test_identical_images(test_images: dict[str, Image.Image]) -> None:
    doc1 = Doc(images=[test_images["rgb_red_100"]])
    doc2 = Doc(images=[test_images["rgb_red_100_2"]])
    assert doc1 == doc2


def test_different_images(test_images: dict[str, Image.Image]) -> None:
    doc1 = Doc(images=[test_images["rgb_red_100"]])
    doc2 = Doc(images=[test_images["rgb_blue_100"]])
    assert doc1 != doc2


def test_none_images() -> None:
    doc1 = Doc(images=None)
    doc2 = Doc(images=None)
    assert doc1 == doc2


def test_one_none_image(test_images: dict[str, Image.Image]) -> None:
    doc1 = Doc(images=[test_images["rgb_red_100"]])
    doc2 = Doc(images=None)
    assert doc1 != doc2


def test_different_image_counts(test_images: dict[str, Image.Image]) -> None:
    doc1 = Doc(images=[test_images["rgb_red_100"], test_images["rgb_red_100_2"]])
    doc2 = Doc(images=[test_images["rgb_red_100"]])
    assert doc1 != doc2


def test_different_image_sizes(test_images: dict[str, Image.Image]) -> None:
    doc1 = Doc(images=[test_images["rgb_red_100"]])
    doc2 = Doc(images=[test_images["rgb_red_200"]])
    assert doc1 != doc2


def test_different_image_modes(test_images: dict[str, Image.Image]) -> None:
    doc1 = Doc(images=[test_images["rgb_red_100"]])
    doc2 = Doc(images=[test_images["l_gray_100"]])
    assert doc1 != doc2


def test_doc_comparison_type_error() -> None:
    doc = Doc(images=None)
    with pytest.raises(NotImplementedError):
        doc == 42


def test_docs_from_hf_dataset() -> None:
    """Tests generation of Docs instance from HF dataset."""
    hf_dataset = Dataset.from_dict(
        {"text": ["This is the first document.", "This is the second document."], "label": [0, 1]}
    )
    docs = Doc.from_hf_dataset(hf_dataset)

    assert len(docs) == 2
    assert docs[0].text == "This is the first document."
    assert docs[0].chunks == ["This is the first document."]  # Check post_init
    assert docs[0].id is None
    assert docs[0].uri is None
    assert docs[0].images is None
    assert docs[0].meta == {}
    assert docs[0].results == {}

    assert docs[1].text == "This is the second document."
    assert docs[1].chunks == ["This is the second document."]  # Check post_init

    # Test with a different text column name.
    data_alt_col = {"content": ["Doc A", "Doc B"], "id": ["a", "b"]}
    hf_dataset_alt_col = Dataset.from_dict(data_alt_col)
    docs_alt = Doc.from_hf_dataset(hf_dataset_alt_col, column_map={"text": "content", "id": "id"})
    assert len(docs_alt) == 2
    assert docs_alt[0].text == "Doc A"
    assert docs_alt[1].text == "Doc B"
    assert docs_alt[0].id == "a"
    assert docs_alt[1].id == "b"

    # Test ValueError for missing column.
    with pytest.raises(
        KeyError,
        match=regex.escape("Specified columns '{'wrong_column'}' not found in dataset columns: ['text', 'label']."),
    ):
        Doc.from_hf_dataset(hf_dataset, column_map={"text": "wrong_column"})
