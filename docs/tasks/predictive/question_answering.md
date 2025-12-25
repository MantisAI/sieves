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

---

::: sieves.tasks.predictive.question_answering.core
::: sieves.tasks.predictive.question_answering.bridges
::: sieves.tasks.predictive.schemas.question_answering
