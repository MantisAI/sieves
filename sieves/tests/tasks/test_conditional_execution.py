"""Tests for conditional task execution."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from sieves.data import Doc
from sieves.pipeline import Pipeline
from sieves.tasks.core import Task


class DummyTask(Task):
    """Simple dummy task for testing conditional execution."""

    def __init__(
        self,
        task_id: str | None = None,
        condition: Callable[[Doc], bool] | None = None,
    ):
        """Initialize DummyTask.

        :param task_id: Task ID.
        :param condition: Optional callable that determines whether to process each document.
        """
        super().__init__(
            task_id=task_id or "DummyTask",
            include_meta=False,
            batch_size=-1,
            condition=condition,
        )

    def _call(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        """Process documents by marking them as processed.

        :param docs: Documents to process.
        :return: Processed documents.
        """
        for doc in docs:
            # Mark document as processed
            doc.results[self.id] = {"processed": True}
            yield doc


def test_condition_none_processes_all_docs() -> None:
    """Test that when condition is None, all documents are processed."""
    docs = [
        Doc(text="Short"),
        Doc(text="This is a longer text"),
        Doc(text="Medium length text"),
    ]

    task = DummyTask(condition=None)
    pipe = Pipeline([task])
    result = list(pipe(docs))

    assert len(result) == 3
    # All docs should have been processed
    assert all(doc.results[task.id] is not None for doc in result)


def test_condition_true_for_all_processes_all_docs() -> None:
    """Test that when condition returns True for all docs, all are processed."""
    docs = [
        Doc(text="Short"),
        Doc(text="This is a longer text"),
        Doc(text="Medium length text"),
    ]

    def always_true(doc: Doc) -> bool:
        return True

    task = DummyTask(condition=always_true)
    pipe = Pipeline([task])
    result = list(pipe(docs))

    assert len(result) == 3
    assert all(doc.results[task.id] is not None for doc in result)


def test_condition_false_for_all_skips_all_docs() -> None:
    """Test that when condition returns False for all docs, none are processed."""
    docs = [
        Doc(text="Short"),
        Doc(text="This is a longer text"),
        Doc(text="Medium length text"),
    ]

    task = DummyTask(condition=lambda d: False)
    pipe = Pipeline([task])
    result = list(pipe(docs))

    assert len(result) == 3
    # All docs should have None in results (indicating they were skipped by the condition)
    for doc in result:
        assert doc.results[task.id] is None


def test_condition_mixed_processing() -> None:
    """Test that documents are selectively processed based on condition."""
    docs = [
        Doc(text="Short"),
        Doc(text="This is a much longer text that should be processed"),
        Doc(text="Medium"),
        Doc(text="Another long document that exceeds the short threshold here"),
    ]

    task = DummyTask(condition=lambda d: len(d.text or "") > 20)
    pipe = Pipeline([task])
    result = list(pipe(docs))

    assert len(result) == 4

    # First doc is short - should be skipped
    assert result[0].results[task.id] is None

    # Second doc is long - should be processed
    assert result[1].results[task.id] is not None

    # Third doc is short - should be skipped
    assert result[2].results[task.id] is None

    # Fourth doc is long - should be processed
    assert result[3].results[task.id] is not None


def test_condition_preserves_document_order() -> None:
    """Test that document order is preserved with conditional execution."""
    docs = [Doc(text=f"Doc {i}") for i in range(5)]

    def is_even_index(doc: Doc) -> bool:
        try:
            idx = int(doc.text.split()[-1])
            return idx % 2 == 0
        except (ValueError, IndexError):
            return False

    task = DummyTask(condition=is_even_index)
    pipe = Pipeline([task])
    result = list(pipe(docs))

    # Order should be maintained
    assert len(result) == 5
    for i, doc in enumerate(result):
        assert doc.text == f"Doc {i}"
        assert doc.results['DummyTask'] != i % 2


def test_condition_in_pipeline() -> None:
    """Test that conditions work correctly in multi-task pipelines."""
    docs = [
        Doc(text="short"),
        Doc(text="this is a longer document that should be processed"),
        Doc(text="med"),
    ]

    def is_long(doc: Doc) -> bool:
        return len(doc.text or "") > 15

    # Create two tasks with different conditions
    task1 = DummyTask(task_id="task1", condition=lambda d: len(d.text or "") > 10)
    task2 = DummyTask(task_id="task2", condition=is_long)

    pipe = Pipeline([task1, task2])
    result = list(pipe(docs))

    assert len(result) == 3

    # First doc: short - skipped by both
    assert result[0].results.get("task1") is None
    assert result[0].results.get("task2") is None

    # Second doc: long - processed by both
    assert result[1].results.get("task1") is not None
    assert result[1].results.get("task2") is not None

    # Third doc: medium - skipped by both
    assert result[2].results.get("task1") is None
    assert result[2].results.get("task2") is None


def test_condition_none_result_storage() -> None:
    """Test that skipped documents have None in results."""
    docs = [Doc(text="Test"), Doc(text="Skip this one")]

    def skip_second(doc: Doc) -> bool:
        return doc.text != "Skip this one"

    task = DummyTask(condition=skip_second)
    pipe = Pipeline([task])
    result = list(pipe(docs))

    # First doc processed
    assert result[0].results[task.id] is not None
    # Second doc skipped
    assert result[1].results[task.id] is None

    # But order is preserved
    assert result[0].text == "Test"
    assert result[1].text == "Skip this one"


def test_condition_exception_handling() -> None:
    """Test that exceptions in condition functions are properly raised."""
    docs = [Doc(text="Test")]

    def bad_condition(doc: Doc) -> bool:
        raise ValueError("Intentional error")

    task = DummyTask(condition=bad_condition)
    pipe = Pipeline([task])

    try:
        list(pipe(docs))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Intentional error" in str(e)


def test_condition_with_task_id() -> None:
    """Test that conditions work with custom task IDs."""
    docs = [
        Doc(text="short"),
        Doc(text="this is a much longer text that will be processed"),
    ]

    def is_long(doc: Doc) -> bool:
        return len(doc.text or "") > 20

    # Use task with custom ID and condition
    task = DummyTask(task_id="custom_task", condition=is_long)
    pipe = Pipeline([task])
    result = list(pipe(docs))

    assert len(result) == 2

    # First doc is short - skipped
    assert result[0].results["custom_task"] is None

    # Second doc is long - processed
    assert result[1].results["custom_task"] is not None


def test_multiple_docs_same_text_with_condition() -> None:
    """Test handling of duplicate docs (same text) with conditions."""
    docs = [
        Doc(text="Duplicate text"),
        Doc(text="Duplicate text"),  # Same text
        Doc(text="Different text"),
    ]

    def skip_duplicates(doc: Doc) -> bool:
        return doc.text != "Duplicate text"

    task = DummyTask(condition=skip_duplicates)
    pipe = Pipeline([task])
    result = list(pipe(docs))

    assert len(result) == 3

    # First two docs skipped (duplicates)
    assert result[0].results[task.id] is None
    assert result[1].results[task.id] is None

    # Third doc processed
    assert result[2].results[task.id] is not None
