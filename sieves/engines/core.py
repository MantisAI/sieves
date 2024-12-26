import abc
from typing import Any, Iterable, Protocol

from sieves.data import Doc
from sieves.tasks import Task


class Engine(Protocol):
    @abc.abstractmethod
    def __call__(self, docs: Iterable[Doc], task: Task, **kwargs: Any) -> Iterable[Doc]:
        """Generate content using the underlying engine.

        :param prompt: The prompt to generate from
        :param kwargs: Additional engine-specific arguments
        :returns: The generated content
        """


class Outlines:
    def __init__(self):
        pass

    def __call__(self, docs: Iterable[Doc], task: Task, **kwargs: Any) -> Iterable[Doc]:
        """Generate content using the underlying engine.

        :param prompt: The prompt to generate from
        :param kwargs: Additional engine-specific arguments
        :returns: The generated content
        """
        raise NotImplementedError
