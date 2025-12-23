# Sentiment Analysis

The `SentimentAnalysis` task determines the sentiment of the text (e.g., positive, negative, neutral).

## Usage

```python
--8<-- "sieves/tests/docs/test_task_usage.py:sentiment-usage"
```

## Results

The `SentimentAnalysis` task returns a unified `Result` object containing a `sentiment_per_aspect` dictionary.

```python
--8<-- "sieves/tasks/predictive/sentiment_analysis/schemas.py:Result"
```

---

::: sieves.tasks.predictive.sentiment_analysis.core
::: sieves.tasks.predictive.sentiment_analysis.bridges
::: sieves.tasks.predictive.sentiment_analysis.schemas
