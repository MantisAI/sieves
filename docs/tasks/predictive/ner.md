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

Each entity includes a confidence score:
- **GLiNER2**: Always present and derived from logits.
- **LLMs**: Self-reported and may be `None` if not provided by the model.

```python
--8<-- "sieves/tasks/predictive/schemas/ner.py:Result"
```

## Evaluation

Performance of the NER task can be measured using the `.evaluate()` method.

- **Metric**: Corpus-wide **Micro-F1 Score** (`F1`). Entities are matched based on their text span (start/end offsets) and type. True Positives, False Positives, and False Negatives are accumulated across the entire dataset.
- **Requirement**: Each document must have ground-truth entities stored in `doc.gold[task_id]`.

```python
report = task.evaluate(docs)
print(f"NER F1-Score: {report.metrics['F1']}")
```

### Ground Truth Formats

Ground truth has to be specified in `doc.meta` using `Result` instances.

---

::: sieves.tasks.predictive.ner.core
::: sieves.tasks.predictive.ner.bridges
::: sieves.tasks.predictive.schemas.ner
