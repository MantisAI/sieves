# Information Extraction

The `InformationExtraction` task allows for structured data extraction from documents using Pydantic schemas.

## Usage

!!! note "Fewshot examples and confidence scores"
    If you use few-shot examples and want your model to provided scores, you should include them in your examples. This
    makes it easier for the model to understand what's expected.

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
--8<-- "sieves/tasks/predictive/schemas/information_extraction.py:Result"
```

- `mode="multi"`: Returns a `ResultMulti` object with an `entities` list.
- `mode="single"`: Returns a `ResultSingle` object with a single `entity` (or `None`).

### Confidence Scores

To provide confidence scores for user-defined entity types, Sieves automatically creates a subclass of your provided Pydantic model that includes a `score` field.

The instances returned in the results will have this additional attribute:

```python
class MyEntity(pydantic.BaseModel, frozen=True):
    name: str

# ... execution ...

result = doc.results["my_task"].entity
print(result.name)
print(result.score)  # Confidence score between 0 and 1, or None for some LLM outputs
```

While confidence scores are always present for **GLiNER2** models, they are self-reported and optional for **LLMs** (DSPy, Outlines, LangChain).

If your original model already contains a `score` field, Sieves will use it as-is without further modification.

---

::: sieves.tasks.predictive.information_extraction.core
::: sieves.tasks.predictive.information_extraction.bridges
::: sieves.tasks.predictive.schemas.information_extraction
