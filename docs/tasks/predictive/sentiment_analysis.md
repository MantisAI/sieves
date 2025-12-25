# Sentiment Analysis

The `SentimentAnalysis` task determines the sentiment of the text (e.g., positive, negative, neutral).

## Usage

```python
--8<-- "sieves/tests/docs/test_task_usage.py:sentiment-usage"
```

## Results

The `SentimentAnalysis` task returns a unified `Result` object containing a `sentiment_per_aspect` dictionary and an overall confidence `score`.

Confidence scores are self-reported by **LLMs** and may be `None`.

## Evaluation

Performance of sentiment analysis can be measured using the `.evaluate()` method.

- **Metric**: Average similarity (1 - absolute error) across all sentiment aspects.
- **Requirement**: Each document must have ground-truth sentiment scores stored in `doc.gold[task_id]`.

```python
report = task.evaluate(docs)
print(f"Sentiment Score: {report.metrics['score']}")
```

---


::: sieves.tasks.predictive.sentiment_analysis.core
::: sieves.tasks.predictive.sentiment_analysis.bridges
::: sieves.tasks.predictive.schemas.sentiment_analysis
