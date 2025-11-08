"""Fallback ingestion task when optional dependencies are unavailable."""

from collections.abc import Iterable
from typing import Any, override

from sieves.data import Doc
from sieves.tasks.core import Task

Converter = Any


class MissingIngestion(Task):
    """Placeholder task raised when ingestion backends are missing.

    Instantiating this class informs users to install the `ingestion` extra.
    """

    @override
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        raise ImportError("Optional ingestion dependency not installed. Install with: uv sync --extra ingestion")

    @override
    def _call(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        raise NotImplementedError
