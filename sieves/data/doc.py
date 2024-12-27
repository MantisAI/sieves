import dataclasses
from typing import Any, Optional


@dataclasses.dataclass
class Doc:
    """A document holding data to be processed."""

    content: str
    chunks: Optional[list[str]]
    meta: dict[str, Any]
    id: Optional[str] = None

    def __str__(self) -> str:
        return self.content
