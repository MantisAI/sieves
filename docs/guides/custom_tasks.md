# Creating Custom Tasks

This guide explains how to create custom tasks. `sieves` distinguishes two types of tasks:
1. Ordinary tasks inherit from `Task`. Pretty much their only requirement is to process a bunch of documents and output
   the same set of documents with their modifications.
2. Predictive tasks inherit from `PredictiveTask` (which inherits from `Task`). Those are for tasks using engines (i.e.
   zero-shot models). They are more complex, as they need to implement the required interface to integrate with at least
   one engines.

While there are a bunch of pre-built tasks available for you to use, you might want to write your own to match your
use-case. This guide describes how to do that.

If you feel like your task might be useful for others, we'd happy to see you submit a PR!

## Tasks

Inherit from `Task` whenever you want to implement something that doesn't require interacting with engines.
That can be document pre- or postprocessing, or something completely different - you could e.g. run an agent following
instructions provided in `docs`, and then follow this up with a subsequent task in your pipeline analyzing and
structuring those results.

To create a basic custom task, inherit from the `Task` class and implement the required abstract methods. In this case
we'll implement a dummy task that counts how many characters are in the document's text and stores that as a result.

```python title="Basic custom task"
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-task-basic"
```

That's it! You can customize this, of course. You might also want to extend `__init__()` to allow for initializing what
you need.

## Predictive Tasks

Inherit from `PredictiveTask` whenever you want to make use of the structured generation capabilities in `sieves`.
`PredictiveTask` requires you to implement a few methods that define how your task expects results to be structured, how
few-shot examples are expected to look like, which prompt to use etc.

We'll break down how to create a predictive task step by step. For this example, let's implement a sentiment analysis
task using `outlines`.

### 1. Implement a `Bridge`

A `Bridge` defines how to solve a task for a certain engine. We decided to go with `outlines` as our engine (you can
allow multiple engines for a task by implementing corresponding bridges, but for simplicity's sake we'll stick with
DSPy only here).

A `Bridge` requires you to implement/specify the following:
- A _prompt template_ (optional depending on the engine used).
- A _prompt signature description_ (optional depending on the engine used).
- A _prompt signature_ describing how results have to be structured.
- How to _integrate_ results into docs.
- How to _consolidate_ results from multiple doc chunks into one result per doc.

The _inference mode_ (which defines how the engine queries the model and parses the results) is configured via `GenerationSettings` when creating the task, rather than in the Bridge.

We'll save this in `sentiment_analysis_bridges.py`.

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-bridge-sentiment"
```

Our bridge takes care of most of the heavy lifting: it defines how we expect our results to look like,
it consolidates the results we're getting back from the engine, and integrates them into our docs.

### 2. Build a `SentimentAnalysisTask`

The task class itself is mostly glue code: we instantiate our bridge(s) and provide other auxiliary, engine-agnostic
functionality. We'll save this in `sentiment_analysis_task.py`

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-task-predictive"
```

And that's it! Our sentiment analysis task is finished.

### 3. Running our task

We can now use our sentiment analysis task like every built-in task:

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-task-usage"
```
