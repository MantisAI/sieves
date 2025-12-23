# Relation Extraction

The `RelationExtraction` task performs joint entity and relation extraction, identifying relationships between entities in text.

## Usage

```python
--8<-- "sieves/tests/tasks/predictive/test_relation_extraction.py:re-usage"
```

## Results

The `RelationExtraction` task returns a unified `Result` object containing a list of `RelationTriplet` objects.

```python
--8<-- "sieves/tasks/predictive/schemas/relation_extraction.py:Result"
```

Each `RelationTriplet` consists of:
- `head`: A `RelationEntity` representing the subject.
- `relation`: The string identifier of the relationship.
- `tail`: A `RelationEntity` representing the object.

A `RelationEntity` includes the surface `text`, `entity_type`, and character `start`/`end` offsets.

---

::: sieves.tasks.predictive.relation_extraction.core
::: sieves.tasks.predictive.relation_extraction.bridges
::: sieves.tasks.predictive.schemas.relation_extraction
