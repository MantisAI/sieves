# mypy: ignore-errors
import time
from collections.abc import Iterable

import pytest

from sieves import Doc, Pipeline, engines, tasks
from sieves.tasks import Classification


@pytest.mark.parametrize(
    "batch_engine",
    [engines.EngineType.outlines],
    indirect=True,
)
def test_double_task(dummy_docs, batch_engine) -> None:
    class DummyTask(tasks.Task):
        def __call__(self, _docs: Iterable[Doc]) -> Iterable[Doc]:
            _docs = list(_docs)
            for _doc in _docs:
                _doc.results[self._task_id] = "dummy"
            yield from _docs

    pipe = Pipeline(
        [
            DummyTask(task_id="task_1", show_progress=False, include_meta=False),
            DummyTask(task_id="task_2", show_progress=False, include_meta=False),
        ]
    )
    docs = list(pipe(dummy_docs))

    _ = pipe["task_1"]
    with pytest.raises(KeyError):
        _ = pipe["sdfkjs"]

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert doc.results["task_1"]
        assert doc.results["task_2"]
        assert "task_1" in doc.results
        assert "task_2" in doc.results


@pytest.mark.parametrize(
    "batch_engine",
    [engines.EngineType.huggingface],
    indirect=True,
)
def test_caching(batch_engine) -> None:
    labels = ["science", "politics"]
    text_science = (
        "Stars are giant balls of hot gas – mostly hydrogen, with some helium and small amounts of other elements. "
        "Every star has its own life cycle, ranging from a few million to trillions of years, and its properties change"
        " as it ages."
    )
    text_politics = (
        "Politics (from Ancient Greek πολιτικά (politiká) 'affairs of the cities') is the set of activities that are "
        "associated with making decisions in groups, or other forms of power relations among individuals, such as the"
        " distribution of status or resources."
    )

    # Test that uniqueness filtering works.

    n_docs = 10
    docs = [Doc(text=text_science) for _ in range(n_docs)]
    pipe = Pipeline(tasks=Classification(labels=labels, engine=batch_engine))
    docs = list(pipe(docs))
    assert pipe._cache_stats == {"hits": 9, "misses": 1, "total": 10, "unique": 1}
    assert len(docs) == n_docs

    # Test that uniqueness filtering works while preserving sequence of Docs.

    docs = [Doc(text=text_science), Doc(text=text_politics), Doc(text=text_science)]
    pipe = Pipeline(tasks=Classification(labels=labels, engine=batch_engine))
    docs = list(pipe(docs))
    assert docs[0].text == docs[2].text == text_science
    assert docs[1].text == text_politics
    assert pipe._cache_stats == {"hits": 1, "misses": 2, "total": 3, "unique": 2}

    # Compare uncached with cached mode with identical documents.

    n_docs = 10
    docs = [Doc(text=text_science) for _ in range(n_docs)]
    uncached_pipe = Pipeline(tasks=Classification(labels=labels, engine=batch_engine), use_cache=False)
    cached_pipe = Pipeline(tasks=Classification(labels=labels, engine=batch_engine))

    start = time.time()
    uncached_docs = list(uncached_pipe(docs))
    uncached_time = time.time() - start

    start = time.time()
    cached_docs = list(cached_pipe(docs))
    cached_time = time.time() - start

    assert len(uncached_docs) == len(cached_docs) == n_docs
    assert cached_pipe._cache_stats == {"hits": 9, "misses": 1, "total": 10, "unique": 1}
    assert uncached_pipe._cache_stats == {"hits": 0, "misses": 10, "total": 10, "unique": 0}
    # Relaxed speed-up requirement: cached pipe should be faster that uncached pipe.
    assert cached_time * 5 < uncached_time

    # Test cache reset.
    cached_pipe.clear_cache()
    assert len(cached_pipe._cache) == 0
    assert cached_pipe._cache_stats == {"hits": 0, "misses": 0, "total": 0, "unique": 0}
