import abc
from typing import Any, Iterable

from sieves.data import Doc
from sieves.interface import Task


class Engine:
    @abc.abstractmethod
    def __call__(self, docs: Iterable[Doc], task: Task, **kwargs: Any) -> Iterable[Doc]:
        """Generate content using the underlying engine.

        :param prompt: The prompt to generate from
        :param kwargs: Additional engine-specific arguments
        :returns: The generated content
        """
        pass
