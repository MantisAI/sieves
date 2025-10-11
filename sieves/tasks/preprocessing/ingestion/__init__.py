"""Ingestion tasks package exports.

Re-export tasks through an import wrapper that gracefully handles optional
dependencies (docling, marker).
"""

from sieves.tasks.preprocessing.ingestion.ingestion_import import Docling, Marker, docling_, marker_  # isort: skip
from sieves.tasks.preprocessing.ingestion.core import Ingestion

__all__ = [
    "Ingestion",
    "Docling",
    "Marker",
    "docling_",
    "marker_",
]
