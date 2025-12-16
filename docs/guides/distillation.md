# Task Distillation

`sieves` supports distilling task results into smaller, specialized models through fine-tuning. This allows you to create fast, efficient models that replicate the behavior of larger zero-shot models without the computational overhead.

## Overview

Distillation in `sieves`:

- **Fine-tunes smaller models** using outputs from zero-shot task execution
- **Reduces inference costs** by replacing expensive LLM calls with lightweight models
- **Maintains performance** while significantly improving speed and reducing resource usage
- **Integrates with popular frameworks**: trains using `setfit` or `model2vec`

The typical workflow is: run a task with a zero-shot LLM → export results → distill to a smaller model → deploy the distilled model for production inference.

## Distillation Workflow

Here's how distillation works in `sieves`:

```
┌────────────────────────────────────────────────────┐
│              DISTILLATION WORKFLOW                  │
└────────────────────────────────────────────────────┘

Step 1: Zero-Shot Inference with Teacher Model
┌────────────────────┐
│   Teacher Model    │  (Large, expensive, accurate)
│   (E.g. big LLM.)  │
└─────────┬──────────┘
          │
          │ Inference: Classify/Extract/Analyze
          │
          ▼
   ┌──────────────────────┐
   │ Training Documents   │
   │ with Model Outputs   │
   │                      │
   │ Doc 1: "AI news"     │
   │   → label: "tech"    │
   │ Doc 2: "Election"    │
   │   → label: "politics"│
   │ ...                  │
   └──────────┬───────────┘
              │
              │ Export to HuggingFace Dataset
              │
              ▼

Step 2: Fine-Tuning Student Model
┌────────────────────────────┐
│   HuggingFace Dataset      │
│ (Text + Teacher Labels)    │
└─────────┬──────────────────┘
          │
          │ Fine-tune with SetFit/Model2Vec
          │
          ▼
   ┌────────────────────┐
   │  Student Model     │  (Small, fast, specialized)
   │ (SetFit/Model2Vec) │
   └─────────┬──────────┘
             │
             │ Inference: Same task, 10-100x faster
             │
             ▼
   ┌────────────────────┐
   │  Production Use    │
   │ (Fast predictions) │
   └────────────────────┘
```

### Performance Characteristics

| Metric | Teacher (LLM) | SetFit Student | Model2Vec Student |
|--------|---------------|----------------|-------------------|
| **Inference Speed** | 1x (baseline) | 50-100x faster | 200-500x faster |
| **Typical Accuracy** | 100% (baseline) | 80-95% retained | 70-85% retained |
| **Model Size** | 10-100GB | 400MB-1GB | 50-100MB |
| **Training Time** | N/A | Minutes | Seconds |

### Why Distillation Works

1. **Teacher provides rich supervision**: The zero-shot model's predictions capture nuanced patterns and edge cases that would be hard to label manually.

2. **Student learns task-specific patterns**: Fine-tuning focuses the smaller model on your exact task, rather than general-purpose language understanding.

3. **Knowledge compression**: The essential decision boundaries are captured in a much smaller parameter space.

4. **Task specialization beats general capability**: A 100M parameter model fine-tuned for sentiment analysis can outperform a 7B parameter general model on that specific task.

## When to Use Distillation

Distillation is valuable when:

- You have **processed documents** with task results from zero-shot models
- You need **faster inference** for production deployment
- You want to **reduce API costs** by avoiding repeated LLM calls
- You're willing to **fine-tune a model** for your specific task


> !!! warning
> Currently, only the **Classification** task has full distillation support via `task.distill()`. Other tasks implement `to_hf_dataset()` for exporting results to Hugging Face datasets, allowing custom training workflows.

## Choosing a Distillation Framework

`sieves` supports two distillation frameworks, each optimized for different scenarios:

### Use SetFit when:

- ✅ **You need good accuracy with limited data** (50-500 examples)
- ✅ **Inference speed is important but not critical** (10-100x faster than LLM)
- ✅ **You can afford ~1GB model size**
- ✅ **You want mature, well-tested framework** (built on sentence-transformers)

### Use Model2Vec when:

- ✅ **Inference speed is critical** (need 100x+ faster than LLM)
- ✅ **Memory is constrained** (<100MB models preferred)
- ✅ **You have sufficient training data** (500+ examples recommended)
- ✅ **Slight accuracy loss is acceptable**
- ✅ **You need extreme efficiency** (CPU-only deployment, edge devices)

### When to skip distillation:

- ❌ **You have <50 training examples per class** - Not enough data for reliable student model
- ❌ **Inference speed isn't a bottleneck** - Just use the teacher model directly
- ❌ **Model accuracy is paramount** - Teacher model will always be more accurate
- ❌ **Teacher model is already small** - Distillation overhead may not be worth it

### Quick Comparison Table

| Aspect                 | SetFit                 | Model2Vec                |
|------------------------|------------------------|--------------------------|
| **Min data**           | 50-100 examples        | 500+ examples            |
| **Model size**         | ~400MB-1GB             | ~50MB-100MB              |
| **Training time**      | Minutes                | Seconds                  |
| **Best for**           | General classification | Extreme speed/efficiency |

## Quick Example

Here's a step-by-step guide to distilling a classification task using `setfit`.

### 1. Import Dependencies

Start by importing the required modules for distillation:

```python
--8<-- "sieves/tests/docs/test_distillation.py:distillation-setfit-imports"
```

These imports provide the task classes, distillation framework, and `setfit` model loader needed for the complete distillation workflow.

### 2. Prepare Training Data

With our dependencies ready, let's create labeled training documents. SetFit requires at least 3 examples per label, but more examples (50-100+) will produce better results:

```python
--8<-- "sieves/tests/docs/test_distillation.py:distillation-setfit-data"
```

Each document includes the text and a ground truth label in its metadata. These will be used to train the student model to replicate the teacher's behavior.

### 3. Generate Predictions with Teacher Model

Now we'll use a large, powerful teacher model to generate predictions for our training data. These predictions will serve as the training labels for our smaller student model:

```python
--8<-- "sieves/tests/docs/test_distillation.py:distillation-setfit-teacher"
```

The teacher processes all documents and stores its predictions in `doc.results`. These predictions capture the teacher's "knowledge" about how to classify these texts.

### 4. Run Distillation

With labeled data from the teacher model, we can now distill this knowledge into a smaller, faster SetFit model. The distillation process will fine-tune a lightweight sentence transformer to replicate the teacher's classification behavior:

```python
--8<-- "sieves/tests/docs/test_distillation.py:distillation-setfit-distill"
```

During distillation, SetFit learns to map text to the same classification decisions as the teacher, but using a much smaller model. The process splits data into train/validation sets (70%/30% here) and trains for the specified number of epochs. The resulting model will be 10-100x faster than the teacher with minimal accuracy loss.

### 5. Load and Use the Distilled Model

Once distillation completes, we can load the student model and use it for fast inference:

```python
--8<-- "sieves/tests/docs/test_distillation.py:distillation-setfit-load"
```

The distilled model is now ready for production use! It provides much faster inference than the teacher model while maintaining most of its accuracy. You can deploy this model anywhere `setfit` is available.

## Distillation Parameters

The `task.distill()` method accepts the following parameters:

```python
task.distill(
    base_model_id: str,              # Hugging Face model ID to fine-tune
    framework: DistillationFramework, # setfit or model2vec
    data: Dataset | Sequence[Doc],   # Documents with task results
    output_path: Path | str,         # Where to save the distilled model
    val_frac: float,                 # Validation set fraction (e.g., 0.2)
    init_kwargs: dict | None = None, # Framework-specific init args
    train_kwargs: dict | None = None,# Framework-specific training args
    seed: int | None = None,         # Random seed for reproducibility
)
```

### Framework-Specific Configuration

**SetFit** (`init_kwargs` and `train_kwargs`):
```python
task.distill(
    base_model_id="sentence-transformers/all-MiniLM-L6-v2",
    framework=DistillationFramework.setfit,
    data=docs,
    output_path="./model",
    val_frac=0.2,
    init_kwargs={
        # Passed to SetFitModel.from_pretrained()
        "multi_target_strategy": "multi-output"  # For multi-label classification
    },
    train_kwargs={
        # Passed to TrainingArguments
        "num_epochs": 3,
        "batch_size": 16,
        "learning_rate": 2e-5,
    }
)
```

**Model2Vec** (`init_kwargs` and `train_kwargs`):
```python
task.distill(
    base_model_id="minishlab/potion-base-8M",
    framework=DistillationFramework.model2vec,
    data=docs,
    output_path="./model",
    val_frac=0.2,
    init_kwargs={
        # Passed to StaticModelForClassification.from_pretrained()
    },
    train_kwargs={
        # Passed to classifier.fit()
        "max_iter": 1000,
    }
)
```

## Using `to_hf_dataset()` for Custom Training

For tasks without built-in distillation support (or for custom training workflows), use `to_hf_dataset()` to export results:

```python
--8<-- "sieves/tests/docs/test_distillation.py:distillation-to-hf-dataset"
```

You can then use this dataset with any training framework like Hugging Face Transformers, SetFit, or custom training loops.

### Threshold Parameter

For classification tasks, `to_hf_dataset()` accepts a `threshold` parameter to convert confidence scores into binary labels:

```python
# Convert multi-label classification results to multi-hot encoding
hf_dataset = classification_task.to_hf_dataset(
    docs,
    threshold=0.5  # Confidences >= 0.5 become 1, others become 0
)
```

## Multi-Label vs Single-Label Classification

The distillation process automatically handles both classification modes:

**Multi-Label** (default):

- Outputs multi-hot boolean vectors
- Each document can have multiple labels
- Uses `multi_target_strategy="multi-output"` for SetFit

**Single-Label**:

- Outputs a single class label
- Each document has exactly one label
- Uses standard classification setup

```python
# Single-label example
task = Classification(
    labels=["technology", "politics", "sports"],
    model=model,
    multi_label=False,
)
```

## Output Structure

After distillation completes, the output directory contains:

```
output_path/
├── data/              # Train/val splits as Hugging Face dataset
├── model files        # Framework-specific model files
└── metrics.json       # Evaluation metrics on validation set
```

**Metrics file** (`metrics.json`):
- `setfit`: Contains F1 score, precision, recall
- `model2vec`: Contains classification metrics

## Best Practices

1. **Use quality zero-shot results**: Distillation quality depends on the quality of your zero-shot predictions
2. **Sufficient data**: Aim for at least 100-500 examples per label for good performance
3. **Validate carefully**: Always check `metrics.json` to ensure distilled model performance is acceptable
4. **Choose appropriate base models**
5. **Split data wisely**: Reserve 20-30% for validation (`val_frac=0.2` is a good default)
6. **Iterate**: If distilled performance is poor, try collecting more diverse examples or using a larger base model

!!! tip "Best Practice: Optimize, Then Distill"
    Start with a small dataset (50-100 examples) to validate your distillation workflow before scaling up. This helps catch configuration issues early without wasting computational resources.

## Troubleshooting

### "Dataset must contain columns: {text, labels}"
- Ensure all documents have results for the task: `doc.results[task_id]` must exist
- If using custom datasets, ensure they have the required columns

### Poor distilled model performance
- Check `metrics.json` in the output directory
- Increase training data (more documents)
- Try a different base model or framework
- Ensure zero-shot predictions are high quality
- Adjust `threshold` parameter for multi-label classification

### Out of memory during training
- Reduce batch size in `train_kwargs`
- Use a smaller base model
- Process documents in smaller batches

### "Unsupported distillation framework for this task"
- Only Classification currently supports distillation via `task.distill()`
- For other tasks, use `to_hf_dataset()` to export results and train manually

## Task Support

| Task                       | `task.distill()`    |
|----------------------------|---------------------|
| **Classification**         | ✅ SetFit, Model2Vec |
| **Sentiment Analysis**     | ❌                   |
| **NER**                    | ❌                   |
| **PII Masking**            | ❌                   |
| **Information Extraction** | ❌                   |
| **Summarization**          | ❌                   |
| **Translation**            | ❌                   |
| **Question Answering**     | ❌                   |

## Related Guides

- **[Task Optimization](optimization.md)** - Optimize tasks before distillation for best student model performance
- **[Custom Tasks](custom_tasks.md)** - Distillation works with custom tasks too (via `to_hf_dataset()`)
- **[Serialization](serialization.md)** - Save distilled model paths in pipeline configurations

!!! tip "Best Practice: Optimize Then Distill"
    For best results, [optimize your task](optimization.md) first to improve the teacher model's accuracy, then distill. A better teacher produces a better student model.

## Further Reading

- [SetFit Documentation](https://huggingface.co/docs/setfit/)
- [Model2Vec Documentation](https://github.com/MinishLab/model2vec)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- [Task-specific documentation](../tasks/predictive/classification.md) for details on each task's output format
