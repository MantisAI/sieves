# Sentiment Analysis

The `SentimentAnalysis` task determines the sentiment of the text (e.g., positive, negative, neutral).

## Usage

```python
--8<-- "sieves/tests/docs/test_task_usage.py:sentiment-usage"
```

## Results

The `SentimentAnalysis` task returns a unified `Result` object containing a `sentiment_per_aspect` dictionary and an overall confidence `score`.

Confidence scores are self-reported by **LLMs** and may be `None`.

```python
--8<-- "sieves/tasks/predictive/schemas/sentiment_analysis.py:Result"
```

---

::: sieves.tasks.predictive.sentiment_analysis.core
::: sieves.tasks.predictive.sentiment_analysis.bridges
::: sieves.tasks.predictive.schemas.sentiment_analysis
