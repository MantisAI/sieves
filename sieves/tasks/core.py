"""Implementations for tasks. Note that there is overlap with sieves.interface.core, as we implement as class hierarchy
for internal use here what we have as protocols (for more flexibility) there. The implementations/class hierarchy
adheres to the interface laid out in sieves.interface.core.
"""

import abc
from typing import Any, Iterable

from sieves.data import Doc


class Task(abc.ABC):
    """Abstract base class for tasks that can be executed on documents."""

    def __init__(self, show_progress: bool = True):
        """
        Initiates new Task.
        :param show_progress: Whether to show progress bar for processed documents.
        """
        self._progress_bar = show_progress

    @abc.abstractmethod
    @property
    def id(self) -> str:
        """Returns task ID. Used by pipeline for results and dependency management.
        :returns: Task ID.
        """


class PreTask(Task):
    @abc.abstractmethod
    def __call__(self, task_input: Any) -> Doc:
        """Parse a set of files.
        :param task_input: Input to process. E.g.: files with `Iterable[sieves.data.File]`.
        :returns: Parsed input in form of documents.
        """


class MainTask(Task):
    @abc.abstractmethod
    def __call__(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        """Execute the task on a set of documents.
        :param docs: The documents to process.
        :returns: The processed document
        """


class PostTask(Task):
    @abc.abstractmethod
    def __call__(self, docs: Iterable[Doc]) -> Any:
        """Execute the task on a set of documents.
        :param docs: The documents to process.
        :returns: The result of the task, which can be any kind of object.
        """
