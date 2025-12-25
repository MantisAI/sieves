# Summarization

The `Summarization` task generates concise summaries of the documents.

## Usage

```python
--8<-- "sieves/tests/docs/test_task_usage.py:summarization-usage"
```

## Results

The `Summarization` task returns a unified `Result` object containing the `summary` and a confidence `score`.

Confidence scores are self-reported by **LLMs** and may be `None`.

```python
--8<-- "sieves/tasks/predictive/schemas/summarization.py:Result"
```

---

::: sieves.tasks.predictive.summarization.core
::: sieves.tasks.predictive.summarization.bridges
::: sieves.tasks.predictive.schemas.summarization
