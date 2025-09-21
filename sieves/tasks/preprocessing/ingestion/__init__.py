"""Ingestion task implementation."""

from .core import Ingestion
from .docling_ import Docling
from .marker_ import Marker
from .unstructured_ import Unstructured

__all__ = ["Docling", "Marker", "Ingestion", "Unstructured"]
