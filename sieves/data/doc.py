from __future__ import annotations

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

    def __post_init__(self) -> None:
        if self.chunks is None and self.text is not None:
            self.chunks = [self.text]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Doc):
            raise NotImplementedError
        return (
            self.id == other.id
            and self.uri == other.uri
            and self.text == other.text
            and self.chunks == other.chunks
            and self.results == other.results
        )
