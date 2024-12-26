import abc
from typing import Iterable, Optional, TypeVar

from sieves.data import Doc

ArbitaryTaskInput = TypeVar("ArbitaryTaskInput")
ArbitaryTaskOutput = TypeVar("ArbitaryTaskOutput")


class Task(abc.ABC):
    """Abstract base class for tasks that can be executed on documents."""

    def __init__(self, task_id: Optional[str] = None, show_progress: bool = True):
        """
        Initiates new Task.
        :param task_id: Task ID.
        :param show_progress: Whether to show progress bar for processed documents.
        """
        self._progress_bar = show_progress
        self._task_id = task_id if task_id else str(self.__class__)

    @property
    def id(self) -> str:
        """Returns task ID. Used by pipeline for results and dependency management.
        :returns: Task ID.
        """
        return self._task_id

    @abc.abstractmethod
    @property
    def prompt_template(self) -> str:
        """Returns prompt template in Jinja 2 format.
        :returns: Prompt template in Jinja 2 format.
        """

    @abc.abstractmethod
    def build_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        """Returns fully formed prompts for given docs.
        :returns: Fully formed docs for given prompts.
        """


class PreTask(Task):
    @abc.abstractmethod
    def __call__(self, task_input: ArbitaryTaskInput, **kwargs) -> Doc:
        """Parse a set of files.
        :param task_input: Input to process. E.g.: files with `Iterable[sieves.data.File]`.
        :returns: Parsed input in form of documents.
        """


class MainTask(Task):
    @abc.abstractmethod
    def __call__(self, docs: Iterable[Doc], **kwargs) -> Iterable[Doc]:
        """Execute the task on a set of documents.
        :param docs: The documents to process.
        :returns: The processed document
        """


class PostTask(Task):
    @abc.abstractmethod
    def __call__(self, docs: Iterable[Doc], **kwargs) -> ArbitaryTaskOutput:
        """Execute the task on a set of documents.
        :param docs: The documents to process.
        :returns: The result of the task, which can be any kind of object.
        """
