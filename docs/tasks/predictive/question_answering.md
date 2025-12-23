# Question Answering

The `QuestionAnswering` task answers questions based on the content of the documents.

## Usage

```python
--8<-- "sieves/tests/docs/test_task_usage.py:qa-usage"
```

## Results

The `QuestionAnswering` task returns a unified `Result` object containing a list of `answers` corresponding to the input questions.

```python
--8<-- "sieves/tasks/predictive/question_answering/schemas.py:Result"
```

---

::: sieves.tasks.predictive.question_answering.core
::: sieves.tasks.predictive.question_answering.bridges
::: sieves.tasks.predictive.question_answering.schemas
