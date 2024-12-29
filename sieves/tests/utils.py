import pytest

from sieves import Doc


@pytest.fixture(scope="session")  # type: ignore[misc]
def dummy_docs() -> list[Doc]:
    return [Doc(text="This is a text " * 100)]
