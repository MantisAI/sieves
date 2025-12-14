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

#### Import Dependencies

First, import the required modules for building the bridge:

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-bridge-sentiment-imports"
```

#### Define the Output Schema

Define the structure of results using Pydantic. This specifies both the sentiment score and reasoning:

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-bridge-sentiment-schema"
```

The schema requires a reasoning explanation and a score between 0 and 1.

#### Create the Bridge Class

Start by defining the bridge class that will handle sentiment analysis:

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-bridge-sentiment-class-def"
```

#### Define the Prompt Template

The prompt template uses Jinja2 syntax to support few-shot examples:

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-bridge-sentiment-prompt"
```

This template instructs the model on how to estimate sentiment and allows for optional few-shot examples.

#### Configure Bridge Properties

Define the required properties that configure how the bridge behaves:

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-bridge-sentiment-properties"
```

These properties specify the prompt signature (output structure) and inference mode.

#### Implement Result Integration

The `integrate()` method stores results into documents:

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-bridge-sentiment-integrate"
```

This method extracts the sentiment score from each result and stores it in the document's results dictionary.

#### Implement Result Consolidation

The `consolidate()` method aggregates results from multiple document chunks:

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-bridge-sentiment-consolidate"
```

For sentiment analysis, we compute the average score across chunks and concatenate all reasoning strings.

Our bridge now handles the complete workflow: prompting the model, parsing structured results, integrating them into documents, and consolidating multi-chunk results.

### 2. Build a `SentimentAnalysisTask`

The task class wraps the bridge and provides engine-agnostic functionality. It handles bridge instantiation, few-shot examples, and dataset export. We'll save this in `sentiment_analysis_task.py`.

Since the task needs a working bridge, we'll include the complete implementation here (the bridge from section 1 plus the task wrapper).

#### Import Task Dependencies

Start with the core imports needed for the task:

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-task-predictive-imports"
```

#### Define the Output Schema

Define the sentiment estimation schema:

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-task-predictive-schema"
```

#### Include the Bridge Implementation

Import additional dependencies for the bridge:

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-task-predictive-bridge-imports"
```

Define the bridge class (as shown in section 1):

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-task-predictive-bridge-class"
```

With its prompt template:

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-task-predictive-bridge-prompt"
```

Bridge properties:

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-task-predictive-bridge-properties"
```

And bridge methods for integration and consolidation:

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-task-predictive-bridge-methods"
```

#### Define Few-Shot Example Schema

Define how few-shot examples should be structured:

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-task-predictive-fewshot"
```

This allows users to provide training examples with text and expected sentiment.

#### Create the Task Class

Now create the main task class that uses the bridge:

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-task-predictive-task-class"
```

#### Implement Bridge Initialization

Define how to initialize the bridge for supported engines:

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-task-predictive-init-supports"
```

The task raises an error if an unsupported engine is specified.

#### Add Dataset Export (Optional)

Implement HuggingFace dataset export for analysis or distillation:

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-task-predictive-to-hf-dataset"
```

And that's it! Our sentiment analysis task is complete and ready to use.

### 3. Running our task

We can now use our sentiment analysis task like every built-in task:

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-task-usage"
```
