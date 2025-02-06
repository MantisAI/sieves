# mypy: ignore-errors
from collections.abc import Iterable

import pytest

from sieves import Doc, Pipeline, engines, tasks


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
            return _docs

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
