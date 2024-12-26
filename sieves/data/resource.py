import dataclasses
from pathlib import Path
from typing import Optional

from sieves.data.doc import Doc


@dataclasses.dataclass
class Resource:
    """Object representing resource that is to be/has been parsed."""

    uri: Path | str
    meta: dict[str, any]
    content: Optional[Doc]
