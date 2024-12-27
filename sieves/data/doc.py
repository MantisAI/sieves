import dataclasses
from pathlib import Path
from typing import Any, Optional


@dataclasses.dataclass
class Doc:
    """A document holding data to be processed."""

    uri: Path | str
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)
    results: dict[str, Any] = dataclasses.field(default_factory=dict)
    text: Optional[str] = None
    chunks: Optional[list[str]] = None
    id: Optional[str] = None

    def __str__(self) -> str:
        """String representation of document.
        :returns: Document text or empty string if no text.
        """
        return self.text if self.text else ""
