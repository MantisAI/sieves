import copy
from typing import Iterable, Literal

from interface import CoreTask
from loguru import logger

from sieves.data import Doc
from sieves.interface import Chunker, Engine, PostTask, PreTask, Task


class Pipeline:
    """Executes a sequence of tasks on documents using a specific engine."""

    def __init__(self, engine: Engine, chunker: Chunker, tasks: Iterable[Task] = tuple()):
        """Initialize the pipeline.

        :param engine: The structured generation engine to use.
        :param chunker: Chunker to use.
        :param tasks: List of tasks to execute.
        """
        self._engine = engine
        self._chunker = chunker
        self._tasks = list(tasks)
        self._validate_tasks()

    def add_tasks(self, tasks: Iterable[Task]):
        self._tasks.extend(tasks)
        self._validate_tasks()

    def _validate_tasks(self) -> None:
        """Validates task pipeline to ensure order of tasks is correct. For now this entails only checks on whether:
        (1) preprocessing tasks are before main tasks are before postprocessing tasks,
        (2) a pipeline contains <= 1 preprocessing and <= 1 postprocessing steps.
        """
        stage: Literal["pre", "main", "post"] = "pre"
        task_ids: set[str] = set()

        for i, task in enumerate(self._tasks):
            if isinstance(task, CoreTask):
                stage = "main"
            elif isinstance(task, PreTask):
                if stage != "pre":
                    raise ValueError(
                        "Preprocessing tasks (i.e. tasks with alternative input format for __call__()) can "
                        "only be used once, and have to be the first task in the pipeline."
                    )
                stage = "main"
            else:
                assert isinstance(task, PostTask), ValueError(
                    "Function signatures doen't match. Task has to match one of the following protocols\n:"
                    "- sieves.interface.PreTask\n- sieves.interface.MainTask\n- sieves.interface.PostTask"
                )
                if stage != "main":
                    raise ValueError(
                        "Postprocessing tasks (i.e. tasks with alternative output format for __call__()) can "
                        "only be used once, and have to be the last task in the pipeline."
                    )
                stage = "post"

            if task.id in task_ids:
                raise ValueError("Each task has to have an individual ID. Make sure that's the case.")

    def __call__(self, docs: Iterable[Doc], in_place: bool = False) -> Iterable[Doc]:
        """Process a list of documents through all tasks.

        :param docs: The documents to process.
        :param in_place: Whether to modify docs in place or to create copies.
        :returns: The processed documents.
        """
        processed_docs = docs if in_place else [copy.deepcopy(doc) for doc in docs]

        for i, task in enumerate(self._tasks):
            logger.info(f"Running task {task.id} ({i}/{len(self._tasks)} tasks).")
            processed_docs = task(processed_docs)

        return processed_docs
