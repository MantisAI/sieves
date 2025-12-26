# Question Answering

The `QuestionAnswering` task answers questions based on the content of the documents.

## Usage

```python
--8<-- "sieves/tests/docs/test_task_usage.py:qa-usage"
```

## Results

The `QuestionAnswering` task returns a unified `Result` object containing a list of `qa_pairs`. Each pair couples the input question with its predicted answer and a confidence score.

Confidence scores are self-reported by **LLMs** and may be `None` if the model fails to provide them.

```python
--8<-- "sieves/tasks/predictive/schemas/question_answering.py:Result"
```

## Evaluation

Performance of the Question Answering task is assessed using a "judge" model.

- **Metric**: **LLM Score** (`LLM Score`). A model-based similarity score (0.0 to 1.0) provided by a DSPy judge, averaged across all question-answer pairs.
- **Requirement**: Each document must have ground-truth answers stored in `doc.gold[task_id]`.
- **Judge**: You must provide a `dspy.LM` instance to the `evaluate()` method.

```python
report = task.evaluate(docs, judge=dspy_judge)
print(f"QA Score: {report.metrics['LLM Score']}")
```

### Ground Truth Formats

Ground truth has to be specified in `doc.meta` using `Result` instances.

---

::: sieves.tasks.predictive.question_answering.core
::: sieves.tasks.predictive.question_answering.bridges
::: sieves.tasks.predictive.schemas.question_answering
