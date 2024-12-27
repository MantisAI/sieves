import dataclasses
from typing import Optional


@dataclasses.dataclass
class Doc:
    """A document holding data to be processed."""

    content: str
    chunks: Optional[list[str]]
    meta: dict[str, any]
    id: Optional[str] = None

    def __str__(self):
        return self.content
