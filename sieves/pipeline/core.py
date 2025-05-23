from __future__ import annotations

import copy
import itertools
import typing
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from loguru import logger

from sieves.data import Doc
from sieves.serialization import Attribute, Config, Serializable
from sieves.tasks import Distillation, PredictiveTask, Task


class Pipeline:
    """Pipeline for executing tasks on documents."""

    def __init__(
        self,
        tasks: Iterable[Task] | Task,
        use_cache: bool = True,
    ):
        """Initialize pipeline.
        :param tasks: List of tasks to execute.
        :param use_cache: If True, pipeline will build a cache over processed `Doc`s to ensure that no redundant
            requests will be sent to the model. If False, all `Doc`s will be processed from scratch, regardless of
            whether they have already been processed..
        """
        self._tasks = [tasks] if isinstance(tasks, Task) else list(tasks)
        self._use_cache = use_cache
        self._cache: dict[int, Doc] = {}
        self._cache_stats: dict[str, int] = {"total": 0, "unique": 0, "hits": 0, "misses": 0}
        self._validate_tasks()
        self._set_distillation_targets()

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

    def _set_distillation_targets(self) -> None:
        """Set target task references fpr distillation tasks, if there are any. This is necessary because distillation
        tasks have a lazily initialized required attribute.
        """
        for task in self._tasks:
            if isinstance(task, Distillation):
                target_task = self[task.target_task_id]
                assert issubclass(type(target_task), PredictiveTask)
                task.target_task = typing.cast(PredictiveTask, target_task)  # type: ignore[type-arg]

    def _get_unseen_unique_docs(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        """Yields unseen, unique docs - i.e. those docs that are not in cache and that are unique within the provided
        collection.
        :param docs: Documents to process.
        """
        doc_hashes: set[int] = set()

        for doc in docs:
            assert doc.text or doc.uri
            doc_cache_id = hash(doc.text or doc.uri)

            if doc_cache_id not in self._cache and doc_cache_id not in doc_hashes:
                doc_hashes.add(doc_cache_id)
                self._cache_stats["unique"] += 1
                yield doc

    def __call__(self, docs: Iterable[Doc], in_place: bool = False) -> Iterable[Doc]:
        """Process a list of documents through all tasks.

        :param docs: Documents to process.
        :param in_place: Whether to modify documents in-place or create copies.
        :return Iterable[Doc]: Processed documents.
        """
        docs_iters = itertools.tee(docs if in_place else (copy.deepcopy(doc) for doc in docs), 2)
        processed_docs = self._get_unseen_unique_docs(docs_iters[0]) if self._use_cache else docs_iters[0]

        for i, task in enumerate(self._tasks):
            logger.info(f"Running task {task.id} ({i + 1}/{len(self._tasks)} tasks).")
            processed_docs = task(processed_docs)

        # If returned docs are not iterators (e.g. returned as lists), get corresponding iterators.
        if not isinstance(processed_docs, Iterator):
            processed_docs = iter(processed_docs)

        # Iterate over all docs. Retrieve doc from cache if available, otherwise add to cache.
        for i, doc in enumerate(docs_iters[1]):
            assert doc.text or doc.uri
            self._cache_stats["total"] += 1
            # Docs must either all have URIs or texts. Either is a sufficient identifier. If first task is OCR and not
            # all docs have IDs, pipeline fails. If first task is predictive and not all docs have texts, pipeline
            # fails.
            doc_cache_id = hash(doc.text or doc.uri)

            if doc_cache_id not in self._cache:
                # Update cache.
                self._cache_stats["misses"] += 1
                processed_doc = next(processed_docs)

                if self._use_cache:
                    self._cache[doc_cache_id] = processed_doc

            else:
                self._cache_stats["hits"] += 1
                processed_doc = self._cache[doc_cache_id]

            yield processed_doc

    def dump(self, path: Path | str) -> None:
        """Save pipeline config to disk.
        :param path: Target path.
        """
        self.serialize().dump(path)

    def clear_cache(self) -> None:
        """Clears cache."""
        self._cache.clear()
        self._cache_stats = {k: 0 for k in self._cache_stats}

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
            {
                "tasks": Attribute(value=[task.serialize() for task in self._tasks]),
                "use_cache": Attribute(value=self._use_cache),
            },
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
