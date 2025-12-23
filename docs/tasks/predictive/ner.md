# Named Entity Recognition

The `NER` task identifies and classifies named entities in text.

## Usage

### Simple List of Entities

You can provide a simple list of entity types to extract.

```python
--8<-- "sieves/tests/docs/test_task_usage.py:ner-usage"
```

### Entities with Descriptions (Recommended)

Providing descriptions for each entity type helps the model understand exactly what you are looking for.

```python
--8<-- "sieves/tests/docs/test_task_usage.py:ner-dict-usage"
```

## Results

The `NER` task returns a unified `Result` object (an alias for `Entities`) containing a list of `Entity` objects and the source text.

```python
--8<-- "sieves/tasks/predictive/ner/schemas.py:Result"
```

---

::: sieves.tasks.predictive.ner.core
::: sieves.tasks.predictive.ner.bridges
::: sieves.tasks.predictive.ner.schemas
