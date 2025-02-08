from __future__ import annotations

import copy
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from loguru import logger

from sieves.data import Doc
from sieves.serialization import Attribute, Config, Serializable
from sieves.tasks import Task


class Pipeline:
    """Pipeline for executing tasks on documents."""

    def __init__(
        self,
        tasks: Iterable[Task] | Task,
    ):
        """Initialize pipeline.
        :param tasks: List of tasks to execute.
        """
        self._tasks = [tasks] if isinstance(tasks, Task) else list(tasks)
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

    def __call__(self, docs: Iterable[Doc], in_place: bool = False) -> Iterable[Doc]:
        """Process a list of documents through all tasks.

        :param docs: Documents to process.
        :param in_place: Whether to modify documents in-place or create copies.
        :return Iterable[Doc]: Processed documents.
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
        :return: Serialized pipeline representation.
        """
        return Config.create(
            self.__class__,
            {"tasks": Attribute(value=[task.serialize() for task in self._tasks])},
        )

    @classmethod
    def deserialize(cls, config: Config, tasks_kwargs: Iterable[dict[str, Any]]) -> Pipeline:
        """Generates pipeline from config.
        :param config: Config to generate pipeline from.
        :param tasks_kwargs: Values to inject into task configs. One dict per task (dict can be empty).
        :return: Deserialized pipeline instance.
        """
        config.validate_init_params(cls)
        tasks_kwargs = tuple(tasks_kwargs)

        assert hasattr(config, "tasks")
        assert len(config.tasks.value) == len(tasks_kwargs), ValueError(
            f"len(tasks_kwargs) has to match the number of tasks in this pipeline ({len(config.tasks.value)}."
        )
        assert config.tasks.is_placeholder is False

        # Deserialize tasks.
        tasks: list[Task] = []
        for task_attr, task_kwargs in zip(config.tasks.value, tasks_kwargs):
            # Restore engine config for PredictiveTask config.
            if "engine" in task_attr:
                task_attr["engine"]["value"], engine_cls = Config.from_dict(task_attr["engine"]["value"])
            # Restore task config, if provided as dict.
            match task_attr:
                case dict():
                    task_config, task_cls = Config.from_dict(task_attr)
                case Config():
                    task_config = task_attr
                    task_cls = task_attr.config_cls
                case _:
                    raise TypeError(f"Deserialization can't handle configs of type {type(task_attr)}.")

            # Deserialize task.
            assert issubclass(task_cls, Serializable)
            assert issubclass(task_cls, Task)
            task = task_cls.deserialize(task_config, **task_kwargs)
            tasks.append(task)

        return cls(tasks=tasks)

    def __getitem__(self, task_id: str) -> Task:
        """Gets task with this ID.
        :param task_id: ID of task to fetch.
        :return: Task with specified ID.
        :raises: KeyError if no task with such ID exists.
        """
        for task in self._tasks:
            if task.id == task_id:
                return task

        raise KeyError(f"No task with ID {task_id} exists in this pipeline.")
