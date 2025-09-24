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
t_cls = tasks.predictive.Classification(labels=["science", "politics"], engine=engine)
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

::: sieves.pipeline.core
