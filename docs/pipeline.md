# Pipeline

Pipelines orchestrate sequential execution of tasks and support two ways to define the sequence:

- Verbose initialization using `Pipeline([...])` (allows setting parameters like `use_cache`)
- Succinct chaining with `+` for readability

Examples

```python title="Pipeline creation patterns"
--8<-- "sieves/tests/docs/test_pipeline_docs.py:pipeline-creation-patterns"
```

Note: Ingestion libraries (e.g., `docling`) are optional and not installed by default. Install them manually or via the extra:

```bash
pip install "sieves[ingestion]"
```

## Conditional Task Execution

Tasks support optional conditional execution via the `condition` parameter. This allows you to skip processing certain documents based on custom logic, without materializing all documents upfront.

### Basic Usage

Pass a callable `Condition[[Doc], bool]` to any task to conditionally process documents:

```python title="Basic conditional execution"
--8<-- "sieves/tests/docs/test_pipeline_docs.py:conditional-execution-basic"
```

### Key Behaviors

- **Per-document evaluation**: The condition is evaluated for each document individually
- **Lazy evaluation**: Documents are not materialized upfront; passing documents are batched together for efficient processing
- **Result tracking**: Skipped documents have `results[task_id] = None`
- **Order preservation**: Document order is always maintained, regardless of which documents are skipped
- **No-op when None**: If `condition=None`, all documents are processed

### Multiple Tasks with Different Conditions

Different tasks in a pipeline can have different conditions:

```python title="Multiple conditional tasks"
--8<-- "sieves/tests/docs/test_pipeline_docs.py:conditional-execution-multiple"
```

### Use Cases

- **Skip expensive processing** for documents that don't meet quality criteria
- **Segment processing** by document properties (size, language, format)
- **Optimize pipelines** by processing subsets of data through specific tasks

::: sieves.pipeline.core
