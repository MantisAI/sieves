# Task Optimization

`sieves` supports automatic optimization of task prompts and few-shot examples using [DSPy's MIPROv2](https://dspy-docs.vercel.app/api/optimizers/MIPROv2) optimizer. This can significantly improve task performance when you have labeled data available.

## Overview

Optimization automatically:
- **Refines prompt instructions** to better guide the model
- **Selects optimal few-shot examples** from your dataset
- **Evaluates performance** using task-specific or LLM-based metrics

The process uses Bayesian optimization to find the best combination of prompt and examples that maximizes performance on a validation set.

## When to Use Optimization

### Use optimization when:

- ✅ **You have labeled training data** (10+ examples minimum, 50+ recommended)
- ✅ **Zero-shot performance is suboptimal** (<70% accuracy on your task)
- ✅ **You can invest in API costs** ($5-50 typical per optimization run)
- ✅ **You want to systematically improve prompts** rather than manual trial-and-error
- ✅ **Your task has clear evaluation metrics** (accuracy, F1, etc.)

### Skip optimization when:

- ❌ **You have <10 examples** - Not enough data for reliable optimization
- ❌ **Zero-shot already works well** - Diminishing returns
- ❌ **Budget is tight** - Optimization requires many LLM calls
- ❌ **You need quick prototyping** - Manual few-shot examples are faster initially
- ❌ **Evaluation is subjective** - Hard to automatically measure improvement

### Decision Tree

```
Do you have labeled examples?
├─ No → Collect data first, use zero-shot for now
└─ Yes → How many?
   ├─ <10 examples → Use manual few-shot, don't optimize yet
   ├─ 10-50 examples → Try optimization with minimal settings
   └─ 50+ examples → Optimization recommended
```

### Cost Considerations

> **⚠️ Cost Warning**
> Optimization involves **multiple LLM calls** during the search process. Costs depend on:
>
> - Dataset size (more examples = more evaluations)
> - DSPy optimizer configuration (`num_candidates`, `num_trials`)
> - Model pricing (larger models cost more per call)
>
> **Estimated costs**:
>
> - Small dataset (20 examples), minimal settings: $2-5
> - Medium dataset (100 examples), default settings: $20-50
> - Large dataset (500+ examples), aggressive settings: $100-500
>
> Start with small datasets and conservative optimizer settings to control costs.

## Quick Example

Here's a step-by-step guide to optimizing a classification task.

### 1. Import Dependencies

First, import the required modules for optimization:

```python
--8<-- "sieves/tests/docs/test_optimization.py:optimization-imports"
```

These imports provide the DSPy model, task classes, and the few-shot example schema needed for optimization.

### 2. Prepare Training Data

With our dependencies imported, we'll create labeled examples for the optimizer. Each example needs the input text, expected label, and a confidence score (1.0 for certain labels):

```python
--8<-- "sieves/tests/docs/test_optimization.py:optimization-training-data"
```

The optimizer will use these examples to evaluate different prompt and few-shot combinations. More examples generally lead to better optimization results, but also increase API costs.

### 3. Create the Task

Now that we have training data, let's define the classification task we want to optimize. We'll include label descriptions to help guide the model:

```python
--8<-- "sieves/tests/docs/test_optimization.py:optimization-task-setup"
```

### 4. Configure the Optimizer

With our task defined, we need to set up the optimizer that will search for the best prompt and example combination. The example below uses minimal settings to control API costs during experimentation:

```python
--8<-- "sieves/tests/docs/test_optimization.py:optimization-optimizer-config"
```

The optimizer splits your data into training and validation sets (25% validation here), then uses Bayesian optimization to explore the space of possible prompts and few-shot selections. The minimal settings (`num_candidates=2`, `num_trials=1`) are for cost control during testing - increase these values for more thorough optimization in production.

### 5. Run Optimization

Finally, we execute the optimization process. The optimizer will iteratively test different prompt and example combinations, evaluating each on the validation set:

```python
--8<-- "sieves/tests/docs/test_optimization.py:optimization-run"
```

The optimizer returns two key outputs: the optimized prompt instructions (which may be significantly different from your original prompt) and the selected few-shot examples that were found to maximize performance. You can then use these in your production task for improved accuracy.

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

> **Note**: LLM-based evaluation adds additional costs since each evaluation requires an extra LLM call.

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

> **💡 Cost Control Tip**
> The example above uses minimal settings (`num_candidates=2`, `num_trials=1`) to reduce costs during experimentation. Increase these values for more thorough optimization once you've validated your setup.

## Best Practices

1. **Start small**: Test optimization with 10-20 examples before scaling up
2. **Use conservative settings**: Start with `num_candidates=2` and `num_trials=1`
3. **Monitor costs**: Track API usage, especially with LLM-based evaluation
4. **Split data wisely**: Use 20-30% for validation (`val_frac=0.25` is a good default)
5. **Provide diverse examples**: Include examples covering different edge cases
6. **Consider model choice**: You can use a cheaper model for optimization than for inference

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

## Related Guides

- **[Custom Tasks](custom_tasks.md)** - Create custom tasks that can also be optimized
- **[Task Distillation](distillation.md)** - After optimizing, distill to faster models for production
- **[Serialization](serialization.md)** - Save optimized prompts and examples for reuse

## Learning More About Optimization

`sieves` optimization is built on [DSPy's MIPROv2 optimizer](https://dspy-docs.vercel.app/api/optimizers/MIPROv2). For in-depth guidance on optimization techniques, training data quality, and interpreting results, we recommend exploring these external resources:

### Understanding MIPROv2

- 📖 **[MIPROv2 API Reference](https://dspy-docs.vercel.app/api/optimizers/MIPROv2)** - Core concepts, parameters, and API documentation
- 📖 **[DSPy Optimizers Overview](https://dspy-docs.vercel.app/docs/building-blocks/optimizers)** - Comprehensive guide to DSPy's optimization framework
- 🎓 **[DSPy Optimization Tutorial](https://dspy-docs.vercel.app/docs/tutorials)** - Step-by-step walkthroughs and examples

### Best Practices & Advanced Topics

- 📊 **Training Data Quality** - What makes good training data for optimization (see [DSPy documentation](https://dspy-docs.vercel.app/docs/building-blocks/optimizers#preparing-data))
- 🔍 **Interpreting Results** - Understanding optimizer outputs and evaluating improvements (covered in [DSPy guides](https://dspy-docs.vercel.app/docs/building-blocks/optimizers))
- ⚙️ **Hyperparameter Tuning** - Adjusting `num_trials`, `num_candidates`, and other optimizer settings for better results
- 🎯 **Evaluation Metrics** - Choosing the right metrics for your task (see Evaluation Metrics section above)

### `sieves`-Specific Integration

The main differences when using optimization in `sieves`:

- **Simplified API**: Use `task.optimize(optimizer)` instead of calling DSPy optimizers directly
- **Automatic integration**: Optimized prompts and few-shot examples are automatically integrated into the task
- **Task compatibility**: Works with all `PredictiveTask` subclasses (Classification, NER, InformationExtraction, etc.)
- **Full parameter access**: All DSPy optimizer parameters are available via the `Optimizer` class constructor

For questions specific to `sieves` optimization integration, see the [Troubleshooting](#troubleshooting) section above or consult the [task-specific documentation](../tasks/predictive/classification.md) for evaluation metrics.
