# Classification

The `Classification` task categorizes documents into predefined labels.

## Usage

### Simple List of Labels
...
```python
--8<-- "sieves/tests/docs/test_task_usage.py:classification-dict"
```

## Results

The `Classification` task returns a unified result schema regardless of the model backend used.

```python
--8<-- "sieves/tasks/predictive/schemas/classification.py:Result"
```

- When `multi_label=True` (default): results are of type `ResultMultiLabel`, containing a list of `(label, score)` tuples.
- When `multi_label=False`: results are of type `ResultSingleLabel`, containing a single `label` and `score`.

---

::: sieves.tasks.predictive.classification.core
::: sieves.tasks.predictive.classification.bridges
::: sieves.tasks.predictive.schemas.classification
