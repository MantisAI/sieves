# Information Extraction

The `InformationExtraction` task allows for structured data extraction from documents using Pydantic schemas.

## Usage

### Multi-Entity Extraction (Default)

By default, the task operates in `mode="multi"`, finding all instances of the specified entity.

```python
--8<-- "sieves/tests/docs/test_task_usage.py:ie-multi"
```

### Single-Entity Extraction

Use `mode="single"` when you expect exactly one entity per document (or none). This is useful for summarizing a document into a structured record.

```python
--8<-- "sieves/tests/docs/test_task_usage.py:ie-single"
```

## Results

The `InformationExtraction` task produces unified results based on the chosen mode:

```python
--8<-- "sieves/tasks/predictive/information_extraction/schemas.py:Result"
```

- `mode="multi"`: Returns a `ResultMulti` object with an `entities` list.
- `mode="single"`: Returns a `ResultSingle` object with a single `entity` (or `None`).

---

::: sieves.tasks.predictive.information_extraction.core
::: sieves.tasks.predictive.information_extraction.bridges
::: sieves.tasks.predictive.information_extraction.schemas
