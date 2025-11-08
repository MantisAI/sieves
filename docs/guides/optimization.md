# Task Optimization

`sieves` supports automatic optimization of task prompts and few-shot examples using [DSPy's MIPROv2](https://dspy-docs.vercel.app/api/optimizers/MIPROv2) optimizer. This can significantly improve task performance when you have labeled data available.

## Overview

Optimization automatically:
- **Refines prompt instructions** to better guide the model
- **Selects optimal few-shot examples** from your dataset
- **Evaluates performance** using task-specific or LLM-based metrics

The process uses Bayesian optimization to find the best combination of prompt and examples that maximizes performance on a validation set.

## When to Use Optimization

Optimization is valuable when:
- You have **labeled data** (few-shot examples with ground truth)
- You want to **improve task accuracy** beyond zero-shot performance
- You're willing to invest **time and API cost** for better results

> **⚠️ Cost Warning**
> Optimization involves **multiple LLM calls** during the search process. Costs depend on:
> - Dataset size (more examples = more evaluations)
> - DSPy optimizer configuration (`num_candidates`, `num_trials`)
> - Model pricing (larger models cost more per _call)
>
> Start with small datasets and conservative optimizer settings to control costs.

## Quick Example

Here's how to optimize a classification task:

```python
import os
import dspy
from sieves.tasks import Classification
from sieves.tasks.optimization import Optimizer
from sieves import GenerationSettings

# 1. Create model for optimization
model = dspy.LM("claude-3-haiku-20240307", api_key=os.environ["ANTHROPIC_API_KEY"])

# 2. Create task with few-shot examples (at least 2 required)
examples = [
    Classification.FewshotExampleSingleLabel(
        text="The new AI model achieves state-of-the-art results",
        reasoning="Discusses AI technology advancement",
        label="technology",
        confidence=1.0
    ),
    Classification.FewshotExampleSingleLabel(
        text="Election results show significant voter turnout",
        reasoning="Focuses on electoral process and voting",
        label="politics",
        confidence=1.0
    ),
    # ... add more examples (recommended: 6-20 examples)
]

task = Classification(
    labels=["technology", "politics", "sports"],
    model=model,
    fewshot_examples=examples,
    generation_settings=GenerationSettings(),
)

# 3. Create optimizer
optimizer = Optimizer(
    model=model,                    # Model for optimization (can differ from task model)
    val_frac=0.25,                  # Use 25% of examples for validation
    seed=42,                        # For reproducibility
    shuffle=True,                   # Shuffle data before splitting
    dspy_init_kwargs=dict(
        num_candidates=2,           # Number of prompt candidates to try
        max_errors=3                # Max errors before stopping
    ),
    dspy_compile_kwargs=dict(
        num_trials=1,               # Number of optimization trials
        minibatch=False             # Use full batch (True for large datasets)
    )
)

# 4. Run optimization
best_prompt, best_examples = task.optimize(optimizer, verbose=True)

# The task now uses the optimized prompt and examples automatically
print(f"Optimized prompt: {best_prompt}")
print(f"Number of selected examples: {len(best_examples)}")
```

## Evaluation Metrics

Different tasks use different evaluation approaches:

### Tasks with Specialized Metrics

These tasks have deterministic, task-specific evaluation metrics:

| Task | Metric | Description |
|------|--------|-------------|
| **Classification** | MAE-based accuracy | Mean Absolute Error on confidence scores (multi-label) or exact match (single-label) |
| **Sentiment Analysis** | MAE-based accuracy | Mean Absolute Error across all sentiment aspects |
| **NER** | F1 score | Precision and recall on (entity_text, entity_type) pairs |
| **PII Masking** | F1 score | Precision and recall on (entity_type, text) pairs |
| **Information Extraction** | F1 score | Set-based F1 on extracted entities |

### Tasks with LLM-Based Evaluation

These tasks use a **generic LLM-as-judge evaluator** that compares ground truth to predictions:

- **Summarization** - Evaluates semantic similarity of summaries
- **Translation** - Evaluates translation quality
- **Question Answering** - Evaluates answer correctness

> **Note**: LLM-based evaluation adds additional costs since each evaluation requires an extra LLM _call.

## Optimizer Configuration

The `Optimizer` class accepts several configuration options:

```python
Optimizer(
    model: dspy.LM,              # Model for optimization
    val_frac: float,             # Validation set fraction (e.g., 0.25)
    seed: int | None = None,     # Random seed for reproducibility
    shuffle: bool = True,        # Shuffle data before splitting
    dspy_init_kwargs: dict | None = None,     # DSPy optimizer init args
    dspy_compile_kwargs: dict | None = None,  # DSPy compile args
)
```

### Key DSPy Parameters

**Init kwargs** (passed to MIPROv2 initialization):
- `num_candidates` (default: 10) - Number of prompt candidates per trial
- `max_errors` (default: 10) - Maximum errors before stopping
- `auto` - Automatic prompt generation strategy

**Compile kwargs** (passed to MIPROv2.compile()):
- `num_trials` (default: 30) - Number of optimization trials
- `minibatch` (default: True) - Use minibatch for large datasets
- `minibatch_size` - Size of minibatches when `minibatch=True`

### Cost Control

To minimize costs during experimentation:

```python
# Use cheaper/smaller model for optimization
optimizer = Optimizer(
    model=dspy.LM("gpt-4o-mini"),  # Cheaper model
    val_frac=0.3,                  # Larger validation set
    dspy_init_kwargs=dict(
        num_candidates=2,          # Fewer candidates (faster, cheaper)
        max_errors=3
    ),
    dspy_compile_kwargs=dict(
        num_trials=1,              # Single trial for testing
        minibatch=True,            # Enable minibatching
        minibatch_size=50          # Smaller batches
    )
)
```

## Best Practices

1. **Start small**: Test optimization with 10-20 examples before scaling up
2. **Use conservative settings**: Start with `num_candidates=2` and `num_trials=1`
3. **Monitor costs**: Track API usage, especially with LLM-based evaluation
4. **Split data wisely**: Use 20-30% for validation (`val_frac=0.25` is a good default)
5. **Provide diverse examples**: Include examples covering different edge cases
6. **Consider model choice**: You can use a cheaper model for optimization than for inference

## Example: Multi-Label Classification

```python
import dspy
from sieves.tasks import Classification
from sieves.tasks.optimization import Optimizer

model = dspy.LM("claude-3-haiku-20240307", api_key="...")

# Multi-label examples with per-label confidence
examples = [
    Classification.FewshotExampleMultiLabel(
        text="Quantum computing advances promise faster drug discovery",
        reasoning="Combines technology (quantum computing) and healthcare (drug discovery)",
        confidence_per_label={"technology": 0.9, "healthcare": 0.7, "finance": 0.1}
    ),
    # ... more examples
]

task = Classification(
    labels=["technology", "healthcare", "finance"],
    model=model,
    multi_label=True,
    fewshot_examples=examples
)

optimizer = Optimizer(model=model, val_frac=0.25)
best_prompt, best_examples = task.optimize(optimizer)
```

## Troubleshooting

### "At least two few-shot examples need to be provided"
- Optimization requires a minimum of 2 examples
- Recommended: 6-20 examples for good results

### High costs
- Reduce `num_candidates` and `num_trials`
- Use smaller validation set (but not less than 15% of data)
- Use cheaper model for optimization
- Enable minibatching for large datasets

### Poor performance after optimization
- Ensure examples are diverse and representative
- Check that examples have correct labels/annotations
- Try different `val_frac` values (0.2-0.3 range)
- Increase `num_trials` for more thorough search

## Further Reading

- [DSPy MIPROv2 Documentation](https://dspy-docs.vercel.app/api/optimizers/MIPROv2)
- [DSPy Optimization Guide](https://dspy-docs.vercel.app/docs/building-blocks/optimizers)
- [Task-specific documentation](../tasks/predictive/classification.md) for details on each task's evaluation metric
