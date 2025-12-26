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

- **Metric**: **Macro-averaged F1 Score** (`F1 (Macro)`). This is calculated per aspect across the corpus and then averaged. Continuous sentiment scores (0.0-1.0) are discretized (0 vs 1) for F1 calculation.
- **Requirement**: Each document must have ground-truth sentiment scores stored in `doc.gold[task_id]`.

```python
report = task.evaluate(docs)
print(f"Sentiment Score: {report.metrics['F1 (Macro)']}")
```

### Ground Truth Formats

Ground truth has to be specified in `doc.meta` using `Result` instances.

---


::: sieves.tasks.predictive.sentiment_analysis.core
::: sieves.tasks.predictive.sentiment_analysis.bridges
::: sieves.tasks.predictive.schemas.sentiment_analysis
