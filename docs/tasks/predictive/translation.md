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

## Evaluation

Evaluation of translations is performed using a "judge" model to measure semantic overlap.

- **Metric**: **LLM Score** (`LLM Score`). A model-based similarity score (0.0 to 1.0) provided by a DSPy judge.
- **Requirement**: Each document must have a ground-truth translation stored in `doc.gold[task_id]`.
- **Judge**: You must provide a `dspy.LM` instance to the `evaluate()` method.

```python
# Evaluate translation quality using a judge
report = task.evaluate(docs, judge=dspy_judge)
print(f"Translation Accuracy: {report.metrics['LLM Score']}")
```

### Ground Truth Formats

Ground truth has to be specified in `doc.meta` using `Result` instances.

---

::: sieves.tasks.predictive.translation.core
::: sieves.tasks.predictive.translation.bridges
::: sieves.tasks.predictive.schemas.translation
