import copy
import inspect
from typing import Any, Callable, Iterable, Optional, Tuple, Type, get_args, get_origin

from loguru import logger
from tasks.core import TaskInput, TaskOutput

from sieves.data import Doc
from sieves.tasks import Task


class Pipeline:
    """Pipeline for executing tasks on documents."""

    def __init__(
        self,
        tasks: Iterable[Task[TaskInput, TaskOutput]],
    ):
        """Initialize the pipeline.
        :param tasks: List of tasks to execute.
        """
        self._tasks = list(tasks)
        self._validate_tasks()

    def add_tasks(self, tasks: Iterable[Task[TaskInput, TaskOutput]]) -> None:
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
        prev_task_sig: Optional[tuple[list[Type[Any]], list[Type[Any]]]] = None

        for i, task in enumerate(self._tasks):
            # Ensure that call return signature of previous task matches function argument signature of current one.
            task_sig = Pipeline._extract_signature_types(task.__call__)
            if prev_task_sig and (prev_task_sig[1] != task_sig[0]):
                raise ValueError(
                    f"Task {task_ids[-1]} has return type {prev_task_sig[1]}, next task {task.id} has input types "
                    f"{task_sig[0]}. These types don't match. Ensure that subsequent tasks have matching output and "
                    f"input types."
                )
            prev_task_sig = task_sig

            if task.id in task_ids:
                raise ValueError("Each task has to have an individual ID. Make sure that's the case.")
            task_ids.append(task.id)

    @staticmethod
    def _extract_signature_types(fn: Callable[..., Any]) -> tuple[list[Type[Any]], list[Type[Any]]]:
        """Extract type of first function argument and return annotation.
        :param fn: Callable to inspect.
        :returns: (1) Types of arguments, (2) types of return annotation (>= 1 if it's a tuple).
        :raises: TypeError if function has more than one argument (this isn't permissible within the currently
        supported architecture).
        """
        sig = inspect.signature(fn)

        def _extract_types(annotation: Type[Any]) -> list[Type[Any]]:
            # Check if it's a tuple type (either typing.Tuple or regular tuple)
            origin = get_origin(annotation)
            if origin is tuple or origin is Tuple:
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
