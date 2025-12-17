# mypy: ignore-errors
import time
from collections.abc import Iterable
import tqdm

import pytest

from sieves import Doc, Pipeline, model_wrappers, tasks
from sieves.tasks import Classification


@pytest.mark.parametrize(
    "batch_runtime",
    [model_wrappers.ModelType.outlines],
    indirect=True,
)
def test_double_task(dummy_docs, batch_runtime) -> None:
    class DummyTask(tasks.Task):
        def _call(self, _docs: Iterable[Doc]) -> Iterable[Doc]:
            _docs = list(_docs)
            for _doc in _docs:
                _doc.results[self._task_id] = "dummy"
            yield from _docs

    pipe = Pipeline(
        [
            DummyTask(task_id="task_1", include_meta=False, batch_size=1),
            DummyTask(task_id="task_2", include_meta=False, batch_size=1),
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
    "batch_runtime",
    [model_wrappers.ModelType.huggingface],
    indirect=True,
)
def test_caching(batch_runtime) -> None:
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
    pipe = Pipeline(tasks=Classification(labels=labels, model=batch_runtime.model, model_settings=batch_runtime.model_settings, batch_size=batch_runtime.batch_size))
    docs = list(pipe(docs))
    assert pipe._cache_stats == {"hits": 9, "misses": 1, "total": 10, "unique": 1}
    assert len(docs) == n_docs

    # Test that uniqueness filtering works while preserving sequence of Docs.

    docs = [Doc(text=text_science), Doc(text=text_politics), Doc(text=text_science)]
    pipe = Pipeline(tasks=Classification(labels=labels, model=batch_runtime.model, model_settings=batch_runtime.model_settings, batch_size=batch_runtime.batch_size))
    docs = list(pipe(docs))
    assert docs[0].text == docs[2].text == text_science
    assert docs[1].text == text_politics
    assert pipe._cache_stats == {"hits": 1, "misses": 2, "total": 3, "unique": 2}

    # Compare uncached with cached mode with identical documents.

    n_docs = 10
    docs = [Doc(text=text_science) for _ in range(n_docs)]
    uncached_pipe = Pipeline(tasks=Classification(labels=labels, model=batch_runtime.model, model_settings=batch_runtime.model_settings, batch_size=batch_runtime.batch_size), use_cache=False)
    cached_pipe = Pipeline(tasks=Classification(labels=labels, model=batch_runtime.model, model_settings=batch_runtime.model_settings, batch_size=batch_runtime.batch_size))

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
    # This can be a bit flaky, but 3x is usually on the safer side.
    assert cached_time * 3 < uncached_time

    # Test cache reset.
    cached_pipe.clear_cache()
    assert len(cached_pipe._cache) == 0
    assert cached_pipe._cache_stats == {"hits": 0, "misses": 0, "total": 0, "unique": 0}


def test_model_wrapper_imports() -> None:
    """Tests direct runtime imports."""
    from sieves.model_wrappers import DSPy, GliNER, HuggingFace, LangChain, Outlines  # noqa: F401


def test_add_task_task(dummy_docs) -> None:
    """Chaining two tasks with ``+`` yields a working Pipeline with both results."""

    class DummyTask(tasks.Task):
        def _call(self, _docs: Iterable[Doc]) -> Iterable[Doc]:
            _docs = list(_docs)
            for _doc in _docs:
                _doc.results[self._task_id] = "ok"
            yield from _docs

    pipe = (
        DummyTask(task_id="t1", include_meta=False, batch_size=-1)
        + DummyTask(task_id="t2", include_meta=False, batch_size=-1)
    )

    assert isinstance(pipe, Pipeline)
    docs = list(pipe(dummy_docs))
    assert len(docs) == 2
    for d in docs:
        assert d.results["t1"] == "ok"
        assert d.results["t2"] == "ok"


def test_add_pipeline_task_and_task_pipeline(dummy_docs) -> None:
    """Chaining Pipeline+Task and Task+Pipeline produces identical outputs order-wise."""

    class DummyTask(tasks.Task):
        def _call(self, _docs: Iterable[Doc]) -> Iterable[Doc]:
            _docs = list(_docs)
            import time
            time.sleep(3)
            for _doc in _docs:
                _doc.results[self._task_id] = "ok"
            yield from _docs

    t1 = DummyTask(task_id="t1", include_meta=False, batch_size=1)
    t2 = DummyTask(task_id="t2", include_meta=False, batch_size=1)

    p1 = Pipeline([t1])
    p2 = p1 + t2
    p3 = t1 + Pipeline([t2])

    for p in (p2, p3):
        docs = list(p(dummy_docs))
        assert len(docs) == 2
        for d in docs:
            assert d.results["t1"] == "ok"
            assert d.results["t2"] == "ok"


if __name__ == '__main__':

    class DummyTask(tasks.Task):
        def _call(self, _docs: Iterable[Doc]) -> Iterable[Doc]:
            _docs = list(_docs)
            import time

            for _doc in _docs:
                time.sleep(.5)
                _doc.results[self._task_id] = "ok"
                yield _doc

    t1 = DummyTask(task_id="t1", include_meta=False, batch_size=1)
    t2 = DummyTask(task_id="t2", include_meta=False, batch_size=1)
    pipe = t1 + t2

    dummy_docs = [Doc(text=str(i)) for i in (1, 2, 3)]
    docs = list(pipe(dummy_docs))
    assert len(docs) == 3

    dummy_docs = [Doc(text=str(i)) for i in (4, 5, 6)]
    for i in tqdm.tqdm([1, 2, 3], desc="Outer loop", total=3, position=0, leave=False):
        docs = list(pipe(dummy_docs))
        assert len(docs) == 3

    dummy_docs = [Doc(text=str(i)) for i in (7, 8, 9)]
    docs = list(pipe(dummy_docs, show_progress=False))
    assert len(docs) == 3


def test_add_pipeline_pipeline(dummy_docs) -> None:
    """Chaining Pipeline+Pipeline concatenates tasks and preserves left cache semantics."""

    class DummyTask(tasks.Task):
        def _call(self, _docs: Iterable[Doc]) -> Iterable[Doc]:
            _docs = list(_docs)
            for _doc in _docs:
                _doc.results[self._task_id] = "ok"
            yield from _docs

    p_left = Pipeline([DummyTask(task_id="left", include_meta=False, batch_size=-1)])
    p_right = Pipeline([DummyTask(task_id="right", include_meta=False, batch_size=-1)])

    p = p_left + p_right
    docs = list(p(dummy_docs))
    for d in docs:
        assert d.results["left"] == "ok"
        assert d.results["right"] == "ok"


def test_add_does_not_mutate_originals() -> None:
    """Chaining should not mutate the original Task or Pipeline instances."""

    class DummyTask(tasks.Task):
        def _call(self, _docs: Iterable[Doc]) -> Iterable[Doc]:
            yield from _docs

    t1 = DummyTask(task_id="t1", include_meta=False, batch_size=-1)
    t2 = DummyTask(task_id="t2", include_meta=False, batch_size=-1)

    p1 = Pipeline([t1])
    p2 = Pipeline([t2])
    p3 = p1 + p2

    assert len(p1._tasks) == 1
    assert len(p2._tasks) == 1
    assert len(p3._tasks) == 2
    assert p3._tasks[0] is t1 and p3._tasks[1] is t2


def test_add_cache_semantics(dummy_docs) -> None:
    """Verify cache propagation rules for all supported chaining combinations."""

    class DummyTask(tasks.Task):
        def _call(self, _docs: Iterable[Doc]) -> Iterable[Doc]:
            yield from _docs

    t1 = DummyTask(task_id="t1", include_meta=False, batch_size=-1)
    t2 = DummyTask(task_id="t2", include_meta=False, batch_size=-1)

    p_uncached = Pipeline([t1], use_cache=False)
    p_cached = Pipeline([t2], use_cache=True)

    # Left pipeline wins
    p = p_uncached + p_cached
    assert p._use_cache is False

    p = p_uncached + t2
    assert p._use_cache is False

    # Task + Pipeline adopts right pipeline cache
    p = t1 + p_cached
    assert p._use_cache is True

    # Task + Task defaults to True
    p = t1 + t2
    assert p._use_cache is True


def test_iadd_pipeline_task(dummy_docs) -> None:
    """Pipeline ``+= Task`` appends in-place and preserves order and cache semantics."""

    class DummyTask(tasks.Task):
        def _call(self, _docs: Iterable[Doc]) -> Iterable[Doc]:
            _docs = list(_docs)
            for _doc in _docs:
                _doc.results[self._task_id] = "ok"
            yield from _docs

    t1 = DummyTask(task_id="t1", include_meta=False, batch_size=-1)
    t2 = DummyTask(task_id="t2", include_meta=False, batch_size=-1)

    p = Pipeline([t1], use_cache=False)
    p += t2

    # Mutated in place
    assert len(p.tasks) == 2
    assert p.tasks[0] is t1 and p.tasks[1] is t2
    assert p.use_cache is False

    docs = list(p(dummy_docs))
    assert len(docs) == 2
    for d in docs:
        assert d.results["t1"] == "ok"
        assert d.results["t2"] == "ok"


def test_iadd_pipeline_pipeline(dummy_docs) -> None:
    """Pipeline ``+= Pipeline`` appends all tasks and preserves left cache semantics."""

    class DummyTask(tasks.Task):
        def _call(self, _docs: Iterable[Doc]) -> Iterable[Doc]:
            _docs = list(_docs)
            for _doc in _docs:
                _doc.results[self._task_id] = "ok"
            yield from _docs

    left = Pipeline([DummyTask(task_id="left", include_meta=False, batch_size=-1)], use_cache=False)
    right = Pipeline([DummyTask(task_id="right", include_meta=False, batch_size=-1)], use_cache=True)

    left += right
    assert len(left.tasks) == 2
    assert left.tasks[0].id == "left" and left.tasks[1].id == "right"
    # Left cache semantics preserved
    assert left.use_cache is False

    docs = list(left(dummy_docs))
    assert len(docs) == 2
    for d in docs:
        assert d.results["left"] == "ok"
        assert d.results["right"] == "ok"
