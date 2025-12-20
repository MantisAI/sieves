# Classification

The `Classification` task categorizes documents into predefined labels.

## Usage

### Simple List of Labels

You can provide a simple list of strings as labels.

```python
--8<-- "sieves/tests/docs/test_task_usage.py:classification-list"
```

### Labels with Descriptions (Recommended)

Providing descriptions for each label helps the model understand the nuances of your classification scheme, often leading to better accuracy.

```python
--8<-- "sieves/tests/docs/test_task_usage.py:classification-dict"
```

---

::: sieves.tasks.predictive.classification.core
::: sieves.tasks.predictive.classification.bridges
