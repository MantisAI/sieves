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

## Evaluation

Performance of the PII masking task can be measured using the `.evaluate()` method.

- **Metric**: Corpus-wide **Micro-F1 Score** (`F1`). PII entities are matched based on their text span (start/end offsets) and type.
- **Requirement**: Each document must have ground-truth PII entities stored in `doc.gold[task_id]`.

```python
report = task.evaluate(docs)
print(f"PII F1-Score: {report.metrics['F1']}")
```

### Ground Truth Formats

Ground truth has to be specified in `doc.meta` using `Result` instances.

---

::: sieves.tasks.predictive.pii_masking.core
::: sieves.tasks.predictive.pii_masking.bridges
::: sieves.tasks.predictive.schemas.pii_masking
