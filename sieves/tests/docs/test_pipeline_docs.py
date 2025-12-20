"""
Test file containing examples for the Pipeline documentation.

These code blocks are referenced in docs/pipeline.md using snippet injection.

Usage in markdown:
    ```python
    --8<-- "sieves/tests/docs/test_pipeline_docs.py:pipeline-creation-patterns"
    ```
"""


def test_pipeline_creation_patterns(example_chunker, small_transformer_model):
    """Test various ways to create and combine pipelines."""
    chunker = example_chunker
    model = small_transformer_model

    # --8<-- [start:pipeline-creation-patterns]
    from sieves import Pipeline, tasks

    # Verbose initialization (allows non-default configuration).
    t_ingest = tasks.preprocessing.Ingestion(export_format="markdown")
    t_chunk = tasks.preprocessing.Chunking(chunker)
    t_cls = tasks.predictive.Classification(labels=["science", "politics"], model=model)
    pipe = Pipeline([t_ingest, t_chunk, t_cls], use_cache=True)

    # Succinct chaining (equivalent task order).
    pipe2 = t_ingest + t_chunk + t_cls

    # You can also chain pipelines and tasks.
    pipe_left = Pipeline(t_ingest)
    pipe_right = Pipeline([t_chunk, t_cls])
    pipe3 = pipe_left + pipe_right  # results in [t_ingest, t_chunk, t_cls]

    # In-place append (mutates the left pipeline).
    # Create new pipeline for demonstration
    pipe_mutable = Pipeline([t_ingest])
    pipe_mutable += t_chunk  # appends t_chunk

    # Can also append entire pipelines
    pipe_to_append = Pipeline([tasks.predictive.Classification(labels=["tech", "sports"], model=model)])
    pipe_mutable += pipe_to_append  # appends all tasks from pipe_to_append

    # Note:
    # - Additional Pipeline parameters (e.g., use_cache=False) are only settable via the verbose form
    # - Chaining never mutates existing tasks or pipelines; it creates a new Pipeline
    # - Using "+=" mutates the existing pipeline by appending tasks
    # --8<-- [end:pipeline-creation-patterns]

    # Assertions for testing
    assert pipe is not None
    assert len(pipe.tasks) == 3


def test_conditional_execution_basic(small_transformer_model):
    """Test basic conditional task execution."""
    model = small_transformer_model

    # --8<-- [start:conditional-execution-basic]
    from sieves import Pipeline, tasks, Doc

    docs = [
        Doc(text="short"),
        Doc(text="this is a much longer document that will be processed"),
        Doc(text="med"),
    ]

    # Define a condition function
    def is_long(doc: Doc) -> bool:
        return len(doc.text or "") > 20

    # Create a task with a condition
    task = tasks.Classification(
        labels=["science", "politics"],
        model=model,
        condition=is_long
    )

    # Run pipeline
    pipe = Pipeline([task])
    for doc in pipe(docs):
        # doc.results[task.id] will be None for documents that failed the condition
        print(doc.results[task.id])
    # --8<-- [end:conditional-execution-basic]

    # Assertions for testing
    results = list(pipe(docs))
    assert len(results) == 3
    # First and third docs should be skipped (too short)
    assert results[0].results[task.id] is None
    assert results[2].results[task.id] is None
    # Second doc should be processed (long enough)
    assert results[1].results[task.id] is not None


def test_conditional_execution_multiple(example_chunker, small_transformer_model):
    """Test multiple tasks with different conditions."""
    chunker = example_chunker
    model = small_transformer_model

    # --8<-- [start:conditional-execution-multiple]
    from sieves import Pipeline, tasks, Doc

    docs = [
        Doc(text="short"),
        Doc(text="this is a much longer document"),
        Doc(text="medium text here"),
    ]

    # Task 1: Process only documents longer than 10 characters
    task1 = tasks.Chunking(chunker, condition=lambda d: len(d.text or "") > 10)

    # Task 2: Process only documents longer than 20 characters
    task2 = tasks.Classification(
        labels=["science", "politics"],
        model=model,
        condition=lambda d: len(d.text or "") > 20
    )

    # First doc: skipped by both tasks (too short)
    # Second doc: processed by both tasks (long enough)
    # Third doc: processed by task1, skipped by task2
    pipe = Pipeline([task1, task2])
    for doc in pipe(docs):
        print(doc.chunks, doc.results[task2.id])
    # --8<-- [end:conditional-execution-multiple]

    # Assertions for testing
    results = list(pipe(docs))
    assert len(results) == 3
    # First doc: skipped by both (condition failed)
    assert results[0].results[task2.id] is None
    # Second doc: processed by both tasks (long enough)
    assert results[1].results[task2.id] is not None
    # Third doc: processed by task1, skipped by task2
    assert results[2].results[task2.id] is None
