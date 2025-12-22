# Creating Custom Tasks

This guide explains how to create custom tasks. `sieves` distinguishes two types of tasks:

1. Ordinary tasks inherit from `Task`. Pretty much their only requirement is to process a bunch of documents and output
   the same set of documents with their modifications.
2. Predictive tasks inherit from `PredictiveTask` (which inherits from `Task`). Those are for tasks using predictive (i.e.
   zero-shot) models. They are more complex, as they need to implement the required interface to integrate with at least
   one model type.

While there are a bunch of pre-built tasks available for you to use, you might want to write your own to match your
use-case. This guide describes how to do that.

If you feel like your task might be useful for others, we'd happy to see you submit a PR!

## Tasks

Inherit from `Task` whenever you want to implement something that doesn't require interacting with models.
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

## Understanding the Architecture

Before diving into implementation, let's understand why `sieves` uses the Bridge pattern and how components fit together.

### The Challenge: Model Diversity

Different NLP frameworks have vastly different APIs:

- **Outlines**: Uses Jinja2 templates + JSON schemas for structured generation
- **DSPy**: Uses signatures + optimizers with Python-native schemas
- **Transformers**: Uses `pipeline()` API with classification/generation modes
- **LangChain**: Uses chains + prompts with custom parsers
- **GliNER2**: Uses structure chaining patterns to describe custom JSON structures

Without abstraction, each task would need separate implementations for each model - unfeasible to maintain.

### The Solution: Bridge Pattern

The Bridge pattern decouples task logic (what to compute) from model-specific implementation (how to compute):

```
┌─────────────────────────────────────────────────────┐
│                     Pipeline                        │
│  (Orchestration: caching, batching, serialization)  │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │        Task          │ ◄── User-facing API
            │  (What to compute)   │     Classification, NER, etc.
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │       Bridge         │ ◄── ModelWrapper-specific adapter
            │  (How to compute)    │     DSPyClassification, OutlinesNER
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │    ModelWrapper      │ ◄── Internal inference handler
            │  (Inference logic)   │     Auto-detected from model type
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │        Model         │ ◄── User-provided.
            │  (3rd party          │     Any supported model
            │   library instance)  │
            └──────────────────────┘
```

### Responsibilities

Each layer has clear, focused responsibilities:

**Task** (User creates this for custom functionality):

- Defines **what** problem to solve (e.g., "classify text into sentiment categories")
- Provides task-specific configuration (labels, entity types, prompt instructions)
- Manages few-shot examples for in-context learning
- Handles optimization (prompt tuning) and distillation (model compression)
- **ModelWrapper-agnostic**: Same task API works with any backend (DSPy, Outlines, etc.)

**Bridge** (User implements this for custom tasks):

- Defines **how** to solve the problem for a specific model type
- Creates prompts (Jinja2 templates for Outlines, signatures for DSPy)
- Parses model outputs into structured results (Pydantic models)
- Integrates results into documents (`integrate()` method)
- Consolidates multi-chunk results into document-level results (`consolidate()` method)
- **ModelWrapper-specific**: Each model type needs its own Bridge implementation

**ModelWrapper** (Internal, automatically detected):

- Handles low-level inference mechanics (batching, generation, error handling)
- Manages model calls and streaming
- Applies model settings (temperature, max_tokens, top_p)
- **Users never interact with this directly** - it's an implementation detail

### Key Methods: integrate() vs consolidate()

These two methods serve different but complementary purposes in the processing pipeline:

**integrate()** - Stores raw results immediately after inference:

```python
def integrate(self, results, docs):
    # Called once per batch of chunks
    # results = [Result1, Result2, Result3]  # One per chunk
    # docs = [Chunk1, Chunk2, Chunk3]        # Corresponding chunks

    for doc, result in zip(docs, results):
        doc.results[self._task_id] = result.score  # Store per-chunk
    return docs
```

**consolidate()** - Merges chunk-level results into document-level results:

```python
def consolidate(self, results: Sequence[TaskResult], docs_offsets: list[tuple[int, int]]) -> Sequence[TaskResult]:
    """Consolidate results for document chunks into document results."""
    # Called after integrate(), once per original document
    # results = [Result1, Result2, Result3]    # All chunk results
    # docs_offsets = [(0, 2), (2, 3)]          # Chunk ranges per doc

    consolidated_results = []
    for start, end in docs_offsets:
        chunk_results = results[start:end]     # Get chunks for this doc
        avg_score = sum(r.score for r in chunk_results) / len(chunk_results)
        consolidated_results.append(ConsolidatedResult(score=avg_score))  # One result per document
    return consolidated_results
```

**Why separate methods?**

- Documents may exceed model context limits (e.g., 100-page PDFs vs 8K token limit)
- `sieves` automatically splits long documents into chunks for processing
- `integrate()` handles per-chunk results (stores immediately, no processing)
- `consolidate()` aggregates chunks back into per-document results (averaging, voting, etc.)

### When to Create Custom Tasks

Use built-in tasks (`Classification`, `NER`, `InformationExtraction`, etc.) whenever possible. Only create custom tasks when:

✅ **Novel task type**: Your task doesn't fit any existing task (e.g., custom scoring, specialized extraction)

✅ **Specialized consolidation**: Your task requires unique logic for merging multi-chunk results (e.g., weighted averaging, consensus voting)

**Don't create custom tasks for:**

❌ **Different prompts**: Use `prompt_instructions` parameter on built-in tasks instead

❌ **Different labels**: Use task configuration (e.g., `labels` parameter in `Classification`)

❌ **Different models**: Just pass a different model instance to the existing task

**Example - When you DON'T need a custom task:**

```python
# ❌ Bad: Creating entire custom task just to change the prompt
class MyCustomClassifier(Classification):
    # 100+ lines of bridge implementation...
    pass

# ✅ Good: Use built-in task with custom prompt
task = Classification(
    labels=["positive", "negative", "neutral"],
    model=model,
    prompt_instructions="Analyze the text's sentiment carefully, "
                        "considering both explicit and implicit cues..."  # Custom!
)
```

Now let's implement a custom task that actually needs custom logic: sentiment analysis with continuous scoring and reasoning.

### 1. Implement a `Bridge`

A `Bridge` defines how to solve a task for a certain model type. We decided to go with `outlines` as our model type (you can
support multiple models for a task by implementing corresponding bridges, but for simplicity's sake we'll stick with
DSPy only here).

A `Bridge` requires you to implement/specify the following:
- A _prompt template_ (optional depending on the model type used).
- A _prompt signature description_ (optional depending on the model type used).
- A _prompt signature_ describing how results have to be structured.
- How to _integrate_ results into docs.
- How to _consolidate_ results from multiple doc chunks into one result per doc.

The _inference mode_ (which defines how the model wrapper queries the model and parses the results) is configured via `ModelSettings` when creating the task, rather than in the Bridge.

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

Start by defining the bridge class that will handle sentiment analysis. In this case we start with a bridge that will
employ Outlines for our sentiment analysis task.

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

The task class wraps the bridge from Section 1 and provides model type-agnostic functionality. It handles bridge instantiation, few-shot examples, and dataset export. We'll save this in `sentiment_analysis_task.py`.

!!! note "Bridge Implementation"
    In a real project, you'd import the bridge from a separate module:
    ```python
    from sentiment_analysis_bridges import OutlinesSentimentAnalysis
    ```

    For this guide's test to be self-contained, the complete code includes both the bridge and task implementation. Below, we show only the task-specific code that's new in this section.

#### Import Task Dependencies

Start with the core imports needed for the task wrapper:

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-task-predictive-imports"
```

These imports provide the base classes and types needed to create a predictive task wrapper.

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

Define how to initialize the bridge for supported models:

```python
--8<-- "sieves/tests/docs/test_custom_tasks.py:custom-task-predictive-init-supports"
```

The task raises an error if an unsupported model type is specified.

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

## Troubleshooting

### Common Issues

#### Results not appearing in `doc.results`

**Symptom**: After running your task, `doc.results[task_id]` is empty or missing.

**Possible causes**:

1. **`integrate()` not called correctly**: Ensure your bridge's `integrate()` method stores results in `doc.results[self._task_id]`
2. **Incorrect task_id**: Verify you're using the correct task ID (check `task._task_id`)
3. **ModelWrapper returning None**: The model wrapper may be returning None results (e.g., due to generation errors in permissive mode)

**Debug steps**:
```python
# Add debug logging to your integrate() method
def integrate(self, results: Sequence[TaskResult], docs: list[Doc]) -> list[Doc]:
    """Integrate results into Doc instances."""
    for doc, result in zip(docs, results):
        if result is None:
            print(f"WARNING: Got None result for doc: {doc.text[:50]}")
        else:
            print(f"Storing result: {result} for task {self._task_id}")
            doc.results[self._task_id] = result
    return docs
```

#### "Model type X is not supported" error

**Cause**: You're trying to use an model type that your bridge doesn't support.

**Solution**: Either:

1. Implement a bridge for that model type
2. Use a supported model type (check `task.supports`)
3. Update `_init_bridge()` to handle the model type

```python
# Check supported models before creating task
from sieves.model_wrappers import ModelType
print(f"Supported models: {task.supports}")  # e.g., {ModelType.outlines}
```

#### Prompt template not rendering correctly

**Symptom**: Model outputs are unexpected or malformed.

**Debug steps**:
1. **Check Jinja2 syntax**: Ensure your template variables are correct
2. **Validate few-shot examples**: Ensure examples match your template's expected structure
3. **Print the rendered prompt**: Add debug logging to see what's actually sent to the model

```python
# In your bridge, add this to see the rendered prompt:
@cached_property
def _prompt_template(self) -> str:
    template = """..."""
    print(f"Template: {template}")
    return template
```

### Best Practices

1. **Start simple**: Begin with a basic bridge, test it, then add complexity
2. **Test consolidate() separately**: Write unit tests with mock data to verify consolidation logic
3. **Handle None results**: Always check for None in integrate() and consolidate()
4. **Use type hints**: Proper typing helps catch errors early
5. **Add assertions**: Use `assert isinstance(result, YourType)` to catch type mismatches
6. **Log generously**: Add debug logging during development to track data flow

## Related Guides

- **[Task Optimization](optimization.md)** - Optimize your custom tasks for better performance
- **[Task Distillation](distillation.md)** - Distill custom tasks using `to_hf_dataset()`
- **[Serialization](serialization.md)** - Save custom tasks (requires providing init_params for complex objects)

!!! tip "Custom Task Serialization"
    When [saving pipelines with custom tasks](serialization.md), you'll need to provide initialization parameters for any complex objects (models, tokenizers, etc.) in `init_params` during load.
