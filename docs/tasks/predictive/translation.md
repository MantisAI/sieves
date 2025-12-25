# Translation

The `Translation` task translates documents into a target language.

## Usage

```python
--8<-- "sieves/tests/docs/test_task_usage.py:translation-usage"
```

## Results

The `Translation` task returns a unified `Result` object containing the `translation` and a confidence `score`.

Confidence scores are self-reported by **LLMs** and may be `None`.

```python
--8<-- "sieves/tasks/predictive/schemas/translation.py:Result"
```

---

::: sieves.tasks.predictive.translation.core
::: sieves.tasks.predictive.translation.bridges
::: sieves.tasks.predictive.schemas.translation
