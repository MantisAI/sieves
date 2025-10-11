"""Import optional ingestion backends with graceful fallbacks.

If a library can't be found, a placeholder task is exported instead.This allows
downstream imports to succeed without hard dependencies.
"""

from __future__ import annotations

import warnings

_missing_dependencies: list[str] = []


try:
    from sieves.tasks.preprocessing.ingestion import docling_  # type: ignore
    from sieves.tasks.preprocessing.ingestion.docling_ import Docling

except (ModuleNotFoundError, ImportError):
    from sieves.tasks.preprocessing.ingestion import missing as docling_  # type: ignore
    from sieves.tasks.preprocessing.ingestion.missing import MissingIngestion as Docling  # type: ignore

    _missing_dependencies.append("docling")


try:
    from sieves.tasks.preprocessing.ingestion import marker_  # type: ignore
    from sieves.tasks.preprocessing.ingestion.marker_ import Marker

except (ModuleNotFoundError, ImportError):
    from sieves.tasks.preprocessing.ingestion import missing as marker_  # type: ignore
    from sieves.tasks.preprocessing.ingestion.missing import MissingIngestion as Marker  # type: ignore

    _missing_dependencies.append("marker")


if _missing_dependencies:
    warnings.warn(
        "Warning: ingestion dependencies [{deps}] could not be imported. Using them requires them to be "
        "installed.".format(deps=", ".join(_missing_dependencies))
    )


__all__ = [
    "docling_",
    "Docling",
    "marker_",
    "Marker",
]
