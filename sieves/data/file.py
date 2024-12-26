import dataclasses
from pathlib import Path
from typing import Optional

from sieves.data.doc import Doc


@dataclasses.dataclass
class File:
    """Object representing files that are to be/have been parsed."""

    path: Path
    meta: dict[str, any]
    content: Optional[Doc]
