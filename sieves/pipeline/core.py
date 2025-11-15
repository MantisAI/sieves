"""Pipeline."""

from __future__ import annotations

import copy
import itertools
from collections.abc import Iterable, Iterator, Sized
from pathlib import Path
from typing import Any

import tqdm

from sieves.data import Doc
from sieves.serialization import Attribute, Config, Serializable
from sieves.tasks import Task


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

    def add_tasks(self, tasks: Iterable[Task]) -> None:
        """Add tasks to pipeline. Revalidates pipeline.

        :param tasks: Tasks to be added.
        """
        self._tasks.extend(tasks)
        self._validate_tasks()

    @property
    def tasks(self) -> list[Task]:
        """Return tasks.

        :return: List of tasks.
        """
        return self._tasks

    @property
    def use_cache(self) -> bool:
        """Return whether pipeline uses cache.

        :return: Whether pipeline uses cache.
        """
        return self._use_cache

    def _validate_tasks(self) -> None:
        """Validate tasks.

        :raises ValueError: On pipeline component signature mismatch.
        """
        task_ids: set[str] = set()

        for i, task in enumerate(self._tasks):
            if task.id in task_ids:
                raise ValueError(f"Task with duplicate ID {task.id}. Ensure unique task IDs.")
            task_ids.add(task.id)

    def _get_unseen_unique_docs(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        """Yield unseen, unique docs.

        I.e. those docs that are not in cache and that are unique within the provided
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

    def __call__(self, docs: Iterable[Doc], in_place: bool = False, show_progress: bool = True) -> Iterable[Doc]:
        """Process a list of documents through all tasks.

        :param docs: Documents to process.
        :param in_place: Whether to modify documents in-place or create copies.
        :parma show_progress: Whether to show progress bar.
        :return Iterable[Doc]: Processed documents.
        """
        n_docs: int | None = len(docs) if isinstance(docs, Sized) else None
        docs_iters = itertools.tee(docs if in_place else (copy.deepcopy(doc) for doc in docs), 2)
        processed_docs = self._get_unseen_unique_docs(docs_iters[0]) if self._use_cache else docs_iters[0]

        for i, task in enumerate(self._tasks):
            processed_docs = task(processed_docs)

        # If returned docs are not iterators (e.g. returned as lists), get corresponding iterators.
        if not isinstance(processed_docs, Iterator):
            processed_docs = iter(processed_docs)

        # Initialize (nested) progress bar, if progress bar is requested.
        progress_bar: tqdm.tqdm | None = None
        if show_progress:
            progress_bar = tqdm.tqdm(desc="Running pipeline", total=n_docs)

        # Iterate over all docs. Retrieve doc from cache if available, otherwise add to cache.
        for i, doc in enumerate(docs_iters[1]):
            assert doc.text or doc.uri
            self._cache_stats["total"] += 1
            # Docs must either all have URIs or texts. Either is a sufficient identifier. If first task is Ingestion
            # and not all docs have IDs, pipeline fails. If first task is predictive and not all docs have texts,
            # pipeline fails.
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

            if show_progress:
                assert progress_bar is not None
                progress_bar.update(1)
                progress_bar.refresh()

            yield processed_doc

        if show_progress:
            assert progress_bar is not None
            progress_bar.close()

    def dump(self, path: Path | str) -> None:
        """Save pipeline config to disk.

        :param path: Target path.
        """
        self.serialize().dump(path)

    def clear_cache(self) -> None:
        """Clear cache."""
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
        """Serialize pipeline object.

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
        """Generate pipeline from config.

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
        """Get task with this ID.

        :param task_id: ID of task to fetch.
        :return: Task with specified ID.
        :raises KeyError: If no task with such ID exists.
        """
        for task in self._tasks:
            if task.id == task_id:
                return task

        raise KeyError(f"No task with ID {task_id} exists in this pipeline.")

    def __add__(self, other: Task | Pipeline) -> Pipeline:
        """Chain this pipeline with another task or pipeline using ``+``.

        Returns a new pipeline that executes all tasks of this pipeline first,
        followed by the task(s) provided via ``other``. The original pipeline(s)
        and task(s) are not mutated.

        Cache semantics:
        - The resulting pipeline preserves this pipeline's ``use_cache`` setting
          regardless of whether ``other`` is a task or pipeline.

        :param other: A ``Task`` or another ``Pipeline`` to execute after this pipeline.
        :return: A new ``Pipeline`` representing the chained execution.
        :raises TypeError: If ``other`` is not a ``Task`` or ``Pipeline``.
        """
        if isinstance(other, Pipeline):
            return Pipeline(tasks=[*self._tasks, *other._tasks], use_cache=self._use_cache)

        if isinstance(other, Task):
            return Pipeline(tasks=[*self._tasks, other], use_cache=self._use_cache)

        raise TypeError(f"Cannot chain Pipeline with {type(other).__name__}")

    def __iadd__(self, other: Task | Pipeline) -> Pipeline:
        """Append a task or pipeline to this pipeline in-place using ``+=``.

        Extending with a pipeline appends all tasks from ``other``. Cache setting
        remains unchanged and follows this (left) pipeline.

        Revalidates the pipeline and updates distillation targets.

        :param other: Task or Pipeline to append.
        :return: This pipeline instance (mutated).
        :raises TypeError: If ``other`` is not a ``Task`` or ``Pipeline``.
        """
        if isinstance(other, Task):
            self._tasks.append(other)
        elif isinstance(other, Pipeline):
            self._tasks.extend(other._tasks)
        else:
            raise TypeError(f"Can only add Task or Pipeline to Pipeline with +=, got {type(other).__name__}")
        self._validate_tasks()

        return self
