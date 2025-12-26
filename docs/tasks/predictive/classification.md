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

- **Metric**: **Macro-averaged F1 Score** (`F1 (Macro)`). This is calculated corpus-wide using `scikit-learn`.
- **Requirement**: Each document must have its ground-truth label stored in `doc.gold[task_id]`.

### Ground Truth Formats

For convenience, you can provide ground-truth data in simplified formats:

- **Single-label (`str`)**: Just the label string.
    ```python
    doc.gold["clf"] = "science"
    ```
- **Multi-label (`list[str]`)**: A list of active labels.
    ```python
    doc.gold["clf"] = ["science", "politics"]
    ```

Alternatively, you can use the standard Pydantic result objects (`ResultSingleLabel`, `ResultMultiLabel`) if you need to specify confidence scores for soft evaluation (though F1 uses hard labels).

```python
report = task.evaluate(docs)
print(f"Classification Score: {report.metrics['F1 (Macro)']}")
```

---

::: sieves.tasks.predictive.classification.core
::: sieves.tasks.predictive.classification.bridges
::: sieves.tasks.predictive.schemas.classification
