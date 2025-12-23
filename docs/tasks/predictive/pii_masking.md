# PII Masking

The `PIIMasking` task identifies and masks Personally Identifiable Information (PII) in documents.

## Usage

```python
--8<-- "sieves/tests/docs/test_task_usage.py:pii-usage"
```

## Results

The `PIIMasking` task returns a unified `Result` object containing the `masked_text` and a list of `pii_entities`.

```python
--8<-- "sieves/tasks/predictive/schemas/pii_masking.py:Result"
```

---

::: sieves.tasks.predictive.pii_masking.core
::: sieves.tasks.predictive.pii_masking.bridges
::: sieves.tasks.predictive.schemas.pii_masking
