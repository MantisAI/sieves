# Pipeline

Pipelines orchestrate sequential execution of tasks and support two ways to define the sequence:

- Verbose initialization using `Pipeline([...])` (allows setting parameters like `use_cache`)
- Succinct chaining with `+` for readability

Examples

```python
from sieves import Pipeline, tasks

# Verbose initialization (allows non-default configuration).
t_ingest = tasks.preprocessing.Ingestion(export_format="markdown")
t_chunk = tasks.preprocessing.Chunking(chunker)
t_cls = tasks.predictive.Classification(labels=["science", "politics"], model=engine)
pipe = Pipeline([t_ingest, t_chunk, t_cls], use_cache=True)

# Succinct chaining (equivalent task order).
pipe2 = t_ingest + t_chunk + t_cls

# You can also chain pipelines and tasks.
pipe_left = Pipeline([t_ingest])
pipe_right = Pipeline([t_chunk, t_cls])
pipe3 = pipe_left + pipe_right  # results in [t_ingest, t_chunk, t_cls]

# In-place append (mutates the left pipeline).
pipe_left += t_chunk
pipe_left += pipe_right  # appends all tasks from right

# Note:
# - Additional Pipeline parameters (e.g., use_cache=False) are only settable via the verbose form
# - Chaining never mutates existing tasks or pipelines; it creates a new Pipeline
# - Using "+=" mutates the existing pipeline by appending tasks
```

Note: Ingestion libraries (e.g., `docling`) are optional and not installed by default. Install them manually or via the extra:

```bash
pip install "sieves[ingestion]"
```

## Conditional Task Execution

Tasks support optional conditional execution via the `condition` parameter. This allows you to skip processing certain documents based on custom logic, without materializing all documents upfront.

### Basic Usage

Pass a callable `Condition[[Doc], bool]` to any task to conditionally process documents:

```python
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
```

### Key Behaviors

- **Per-document evaluation**: The condition is evaluated for each document individually
- **Lazy evaluation**: Documents are not materialized upfront; passing documents are batched together for efficient processing
- **Result tracking**: Skipped documents have `results[task_id] = None`
- **Order preservation**: Document order is always maintained, regardless of which documents are skipped
- **No-op when None**: If `condition=None`, all documents are processed

### Multiple Tasks with Different Conditions

Different tasks in a pipeline can have different conditions:

```python
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
    print(doc.results[task1.id], doc.results[task2.id])
```

### Use Cases

- **Skip expensive processing** for documents that don't meet quality criteria
- **Segment processing** by document properties (size, language, format)
- **Optimize pipelines** by processing subsets of data through specific tasks

::: sieves.pipeline.core
