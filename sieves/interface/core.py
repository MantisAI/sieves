import abc
from typing import Any, Iterable, Protocol

from sieves.data import Doc


class HasID(Protocol):
    @property
    def id(self) -> str:
        """Returns task ID. Used by pipeline for results and dependency management.
        :returns: Task ID.
        """


class PreTask(HasID):
    @abc.abstractmethod
    def __call__(self, task_input: Any) -> Iterable[Doc]:
        """Parse a set of files.
        :param task_input: Input to process. E.g.: files with `Iterable[sieves.data.File]`.
        :returns: Parsed input in form of documents.
        """


class CoreTask(HasID):
    @abc.abstractmethod
    def __call__(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        """Execute the task on a set of documents.
        :param docs: The documents to process.
        :returns: The processed document
        """


class PostTask(HasID):
    @abc.abstractmethod
    def __call__(self, docs: Iterable[Doc]) -> Any:
        """Execute the task on a set of documents.
        :param docs: The documents to process.
        :returns: The result of the task, which can be any kind of object.
        """


Task = PreTask | CoreTask | PostTask


class Engine(Protocol):
    """Protocol defining the interface for structured generation engines."""

    @abc.abstractmethod
    def __call__(self, docs: Iterable[Doc], task: Task, **kwargs: Any) -> Iterable[Doc]:
        """Generate content using the underlying engine.

        :param prompt: The prompt to generate from
        :param kwargs: Additional engine-specific arguments
        :returns: The generated content
        """


class Chunker(abc.ABC):
    """Splits up documents into chunks."""

    @abc.abstractmethod
    def __call__(
        self, docs: Iterable[Doc], include_meta: bool = False
    ) -> Iterable[Iterable[str]] | tuple[Iterable[Iterable[str]], Any]:
        """Splits up documents into chunks.
        :param docs: Documents to chunk.
        :param include_meta: Whether to return meta information in return results.
        :returns: Split up documents or, if include_meta is True, also chunking meta information.
        """
