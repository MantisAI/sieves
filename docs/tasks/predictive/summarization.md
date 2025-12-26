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

## Evaluation

Because summarization is a generative task, evaluation requires a "judge" model to assess semantic similarity.

- **Metric**: **LLM Score** (`LLM Score`). A model-based similarity score (0.0 to 1.0) provided by a DSPy judge.
- **Requirement**: Each document must have a ground-truth summary stored in `doc.gold[task_id]`.
- **Judge**: You must provide a `dspy.LM` instance to the `evaluate()` method.

```python
# Use an LLM as a judge to evaluate the summaries
report = task.evaluate(docs, judge=dspy_judge)
print(f"Summary Similarity: {report.metrics['LLM Score']}")
```

### Ground Truth Formats

Ground truth has to be specified in `doc.meta` using `Result` instances.

---

::: sieves.tasks.predictive.summarization.core
::: sieves.tasks.predictive.summarization.bridges
::: sieves.tasks.predictive.schemas.summarization
