from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, get_args, get_origin

from loguru import logger

from sieves.data import Doc
from sieves.serialization import Attribute, Config, Serializable
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
        task_ids: set[str] = set()

        for i, task in enumerate(self._tasks):
            if task.id in task_ids:
                raise ValueError(f"Task with duplicate ID {task.id}. Ensure unique task IDs.")
            task_ids.add(task.id)

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

    def dump(self, path: Path | str) -> None:
        """Save pipeline config to disk.
        :param path: Target path.
        """
        self.serialize().dump(path)

    @classmethod
    def load(cls, path: Path | str, task_kwargs: Iterable[dict[str, Any]]) -> Pipeline:
        """Generate pipeline from disk.
        :param path: Path to config file.
        :param task_kwargs: Values to inject into loaded config.
        :return: Pipeline instance.
        """
        return cls.deserialize(Config.load(path), task_kwargs)

    def serialize(self) -> Config:
        """Serializes pipeline object.
        :returns: Serialized pipeline representation.
        """
        return Config.create(
            self.__class__,
            {"tasks": Attribute(value=[task.serialize() for task in self._tasks], is_placeholder=False)},
        )

    @classmethod
    def deserialize(cls, config: Config, tasks_kwargs: Iterable[dict[str, Any]]) -> Pipeline:
        """Generates pipeline from config.
        :param config: Config to generate pipeline from.
        :param tasks_kwargs: Values to inject into task configs. One dict per task (dict can be empty).
        :returns: Deserialized pipeline instance.
        """
        config.validate_init_params(cls)
        tasks_kwargs = tuple(tasks_kwargs)

        assert hasattr(config, "tasks")
        assert len(config.tasks.value) == len(tasks_kwargs)
        assert config.tasks.is_placeholder is False

        # Deserialize tasks.
        tasks: list[Task] = []
        for task_attr, task_kwargs in zip(config.tasks.value, tasks_kwargs):
            # Restore engine config for PredictiveTask config.
            if "engine" in task_attr:
                task_attr["engine"]["value"], engine_cls = Config.from_dict(task_attr["engine"]["value"])
            # Restore task config.
            task_config, task_cls = Config.from_dict(task_attr)

            # Deserialize task.
            assert issubclass(task_cls, Serializable)
            assert issubclass(task_cls, Task)
            task = task_cls.deserialize(task_config, **task_kwargs)
            tasks.append(task)

        return cls(tasks=tasks)
