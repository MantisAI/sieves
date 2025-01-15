import copy
import inspect
from collections.abc import Callable, Iterable
from typing import Any, get_args, get_origin

from loguru import logger

from sieves.data import Doc
from sieves.tasks import Task


class Pipeline:
    """Pipeline for executing tasks on documents."""

    def __init__(
        self,
        tasks: Iterable[Task],
    ):
        """Initialize the pipeline.
        :param tasks: List of tasks to execute.
        """
        self._tasks = list(tasks)
        self._validate_tasks()

    def add_tasks(self, tasks: Iterable[Task]) -> None:
        """Adds tasks to pipeline. Revalidates pipeline.
        :param tasks: Tasks to be added.
        """
        self._tasks.extend(tasks)
        self._validate_tasks()

    def _validate_tasks(self) -> None:
        """Validate tasks.
        :raises: ValueError on pipeline component signature mismatch.
        """
        task_ids: list[str] = []

        for i, task in enumerate(self._tasks):
            if task.id in task_ids:
                raise ValueError("Each task has to have an individual ID. Make sure that's the case.")
            task_ids.append(task.id)

    @staticmethod
    def _extract_signature_types(fn: Callable[..., Any]) -> tuple[list[type[Any]], list[type[Any]]]:
        """Extract type of first function argument and return annotation.
        :param fn: Callable to inspect.
        :returns: (1) Types of arguments, (2) types of return annotation (>= 1 if it's a tuple).
        :raises: TypeError if function has more than one argument (this isn't permissible within the currently
        supported architecture).
        """
        sig = inspect.signature(fn)

        def _extract_types(annotation: type[Any]) -> list[type[Any]]:
            # Check if it's a tuple type (either typing.Tuple or regular tuple)
            origin = get_origin(annotation)
            if origin is tuple or origin is tuple:
                return list(get_args(annotation))
            return [annotation]

        return (
            [param.annotation for param in list(sig.parameters.values()) if param.name != "self"],
            _extract_types(sig.return_annotation),
        )

    def __call__(self, docs: Iterable[Doc], in_place: bool = False) -> Iterable[Doc]:
        """Process a list of documents through all tasks.
        :param docs: Documents to process.
        :param in_place: Whether to modify documents in-place or create copies.
        :returns: Processed documents.
        """
        processed_docs = docs if in_place else [copy.deepcopy(doc) for doc in docs]

        for i, task in enumerate(self._tasks):
            logger.info(f"Running task {task.id} ({i + 1}/{len(self._tasks)} tasks).")
            processed_docs = task(processed_docs)

        return processed_docs
