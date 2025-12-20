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

---

::: sieves.tasks.predictive.ner.core
::: sieves.tasks.predictive.ner.bridges
