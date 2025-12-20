# Task

## Conditional Execution

All tasks support optional conditional execution through the `condition` parameter. This feature allows you to skip processing certain documents based on custom criteria without materializing all documents upfront.

### Overview

The `condition` parameter accepts an optional callable with signature `Callable[[Doc], bool]`:

```python
def condition(doc: Doc) -> bool:
    # Return True to process the document
    # Return False to skip it
    return True
```

### Implementation Details

When a task is executed with a condition:

1. **Per-Document Evaluation**: Each document is evaluated against the condition individually
2. **Lazy Batching**: Only documents that pass the condition are batched together and sent to the task's `_call()` method
3. **Order Preservation**: Documents are returned in their original order, even if some were skipped
4. **Result Storage**: Skipped documents have `results[task_id] = None`

### Examples

#### Skip Documents by Size

```python
from sieves import tasks, Pipeline, Doc

# Only process documents longer than 100 characters
task = tasks.Classification(
    labels={
        "positive": "Positive sentiment or favorable opinion",
        "negative": "Negative sentiment or unfavorable opinion"
    },
    model=model,
    condition=lambda doc: len(doc.text or "") > 100
)

pipe = Pipeline([task])
docs = [Doc(text="short"), Doc(text="a very long document " * 10)]
results = list(pipe(docs))

# First doc: results[task.id] == None (skipped)
# Second doc: results[task.id] contains classification results
```

#### Skip Documents Based on Metadata

```python
# Only process documents from specific sources
def should_process(doc: Doc) -> bool:
    return doc.meta.get("source") in ["source_a", "source_b"]

task = tasks.NER(
    entities={
        "PERSON": "Names of people, including first and last names",
        "LOCATION": "Geographic locations like cities, countries, and landmarks"
    },
    model=model,
    condition=should_process
)
```

#### Multiple Conditions in Pipeline

```python
# Different conditions for different tasks
import_task = tasks.Ingestion(export_format="markdown")

# Only chunk long documents
chunking_task = tasks.Chunking(
    chunker,
    condition=lambda doc: len(doc.text or "") > 500
)

# Only classify chunked documents
classification_task = tasks.Classification(
    labels={
        "science": "Scientific content including research and facts",
        "fiction": "Fictional stories and creative writing"
    },
    model=model,
    condition=lambda doc: len(doc.text or "") > 500
)

pipe = Pipeline([import_task, chunking_task, classification_task])
```

### Technical Notes

- **No Materialization**: Documents are processed using iterators; passing documents are batched together without materializing the entire document collection upfront
- **Index-Based Tracking**: The implementation uses document indices for efficient filtering and reordering
- **All Model wrappers Supported**: Conditional execution works with all supported model libraries (DSPy, LangChain, Outlines, HuggingFace, GLiNER2, etc.)
- **Serialization**: Non-callable condition values (like `None`) serialize naturally; callable conditions are serialized as placeholders

---

::: sieves.tasks.core.Task
