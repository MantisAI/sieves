# Relation Extraction

The `RelationExtraction` task performs joint entity and relation extraction, identifying relationships between entities in text.

## Usage

```python
--8<-- "sieves/tests/tasks/predictive/test_relation_extraction.py:re-usage"
```

## Results

The `RelationExtraction` task returns a unified `Result` object containing a list of `RelationTriplet` objects.

Each triplet includes a confidence score:
- **GLiNER2**: Always present and derived from logits.
- **LLMs**: Self-reported and may be `None` if not provided by the model.

```python
--8<-- "sieves/tasks/predictive/schemas/relation_extraction.py:Result"
```

Each `RelationTriplet` consists of:
- `head`: A `RelationEntity` representing the subject.
- `relation`: The string identifier of the relationship.
- `tail`: A `RelationEntity` representing the object.

A `RelationEntity` includes the surface `text`, `entity_type`, and character `start`/`end` offsets.

## Evaluation

Performance of the relation extraction task can be measured using the `.evaluate()` method.

- **Metric**: Corpus-wide **Micro-F1 Score** (`F1`). Triplets are matched based on the head entity text, the relation type, and the tail entity text.
- **Requirement**: Each document must have ground-truth triplets stored in `doc.gold[task_id]`.

```python
report = task.evaluate(docs)
print(f"Relation F1-Score: {report.metrics['F1']}")
```

### Ground Truth Formats

Ground truth has to be specified in `doc.meta` using `Result` instances.

---

::: sieves.tasks.predictive.relation_extraction.core
::: sieves.tasks.predictive.relation_extraction.bridges
::: sieves.tasks.predictive.schemas.relation_extraction
