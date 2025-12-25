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

- When `mode == 'multi'` (default): results are of type `ResultMultiLabel`, containing a list of `(label, score)` tuples.
- When `mode == 'single'`: results are of type `ResultSingleLabel`, containing a single `label` and `score`.

Confidence scores are always present for `transformers` and `gliner2` models. For **LLMs**, scores are self-reported and may be `None`.

## Evaluation

You can evaluate the performance of your classifier using the `.evaluate()` method.

- **Metric**: For single-label classification, the score is binary (1.0 for a match, 0.0 for a mismatch). For multi-label, the score is the average similarity (1 - absolute error) across all labels.
- **Requirement**: Each document must have its ground-truth label stored in `doc.gold[task_id]`.

```python
report = task.evaluate(docs)
print(f"Classification Score: {report.metrics['score']}")
```

---

::: sieves.tasks.predictive.classification.core
::: sieves.tasks.predictive.classification.bridges
::: sieves.tasks.predictive.schemas.classification
