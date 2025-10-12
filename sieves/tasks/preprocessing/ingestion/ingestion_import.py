"""Import optional ingestion backends with graceful fallbacks.

If a library can't be found, a placeholder task is exported instead.This allows
downstream imports to succeed without hard dependencies.
"""

try:
    from sieves.tasks.preprocessing.ingestion import docling_  # type: ignore
    from sieves.tasks.preprocessing.ingestion.docling_ import Docling

except (ModuleNotFoundError, ImportError):
    from sieves.tasks.preprocessing.ingestion import missing as docling_  # type: ignore
    from sieves.tasks.preprocessing.ingestion.missing import MissingIngestion as Docling  # type: ignore


try:
    from sieves.tasks.preprocessing.ingestion import marker_  # type: ignore
    from sieves.tasks.preprocessing.ingestion.marker_ import Marker

except (ModuleNotFoundError, ImportError):
    from sieves.tasks.preprocessing.ingestion import missing as marker_  # type: ignore
    from sieves.tasks.preprocessing.ingestion.missing import MissingIngestion as Marker  # type: ignore


__all__ = [
    "docling_",
    "Docling",
    "marker_",
    "Marker",
]
