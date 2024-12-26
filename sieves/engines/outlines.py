from typing import Any, Iterable

from sieves.data import Doc
from sieves.engines.core import Engine
from sieves.interface import Task


class Outlines(Engine):
    """Engine implementation using the outlines library."""

    def __init__(self, **kwargs: Any):
        """Initialize the outlines engine.

        :param kwargs: Configuration for the outlines engine
        """
        self.config = kwargs

    def __call__(self, docs: Iterable[Doc], task: Task, **kwargs: Any) -> Iterable[Doc]:
        """Generate content using outlines.

        :param prompt: The prompt to generate from
        :param kwargs: Additional generation parameters
        :returns: The generated content
        """
        # TODO: Implement outlines generation
        raise NotImplementedError
