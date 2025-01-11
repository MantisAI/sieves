import dataclasses
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class Doc:
    """A document holding data to be processed."""

    meta: dict[str, Any] = dataclasses.field(default_factory=dict)
    results: dict[str, Any] = dataclasses.field(default_factory=dict)
    uri: Path | str | None = None
    text: str | None = None
    chunks: list[str] | None = None
    id: str | None = None

    def __str__(self) -> str:
        """String representation of document.
        :returns: Document text or empty string if no text.
        """
        return self.text if self.text else ""
