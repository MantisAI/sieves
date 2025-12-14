# Task Distillation

`sieves` supports distilling task results into smaller, specialized models through fine-tuning. This allows you to create fast, efficient models that replicate the behavior of larger zero-shot models without the computational overhead.

## Overview

Distillation in `sieves`:
- **Fine-tunes smaller models** using outputs from zero-shot task execution
- **Reduces inference costs** by replacing expensive LLM calls with lightweight models
- **Maintains performance** while significantly improving speed and reducing resource usage
- **Integrates with popular frameworks** like SetFit and Model2Vec

The typical workflow is: run a task with a zero-shot LLM → export results → distill to a smaller model → deploy the distilled model for production inference.

## When to Use Distillation

Distillation is valuable when:
- You have **processed documents** with task results from zero-shot models
- You need **faster inference** for production deployment
- You want to **reduce API costs** by avoiding repeated LLM calls
- You're willing to **fine-tune a model** for your specific task

> [!NOTE]
> Currently, only the **Classification** task has full distillation support via `task.distill()`. Other tasks implement `to_hf_dataset()` for exporting results to Hugging Face datasets, allowing custom training workflows.

## Supported Frameworks

`sieves` supports two distillation frameworks:

| Framework | Description | Best For |
|-----------|-------------|----------|
| **SetFit** | Few-shot learning with sentence transformers | General-purpose classification with limited data |
| **Model2Vec** | Static embeddings with lightweight classifiers | Extremely fast inference, minimal memory footprint |

## Quick Example

Here's a step-by-step guide to distilling a classification task using SetFit.

### 1. Import Dependencies

Import the required modules:

```python
--8<-- "sieves/tests/docs/test_distillation.py:distillation-setfit-imports"
```

### 2. Prepare Training Data

Create labeled documents (at least 3 examples per label for SetFit):

```python
--8<-- "sieves/tests/docs/test_distillation.py:distillation-setfit-data"
```

### 3. Generate Predictions with Teacher Model

Define a teacher task and process documents to generate predictions:

```python
--8<-- "sieves/tests/docs/test_distillation.py:distillation-setfit-teacher"
```

The teacher model's predictions will be used as training labels for the student model.

### 4. Run Distillation

Distill the teacher's knowledge into a smaller, faster model:

```python
--8<-- "sieves/tests/docs/test_distillation.py:distillation-setfit-distill"
```

This trains a lightweight SetFit model that mimics the teacher's behavior.

### 5. Load and Use the Distilled Model

Load the distilled model and use it for inference:

```python
--8<-- "sieves/tests/docs/test_distillation.py:distillation-setfit-load"
```

The distilled model is now ready for production use with faster inference and lower resource requirements.

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
- SetFit: Contains F1 score, precision, recall
- Model2Vec: Contains classification metrics

## Best Practices

1. **Use quality zero-shot results**: Distillation quality depends on the quality of your zero-shot predictions
2. **Sufficient data**: Aim for at least 100-500 examples per label for good performance
3. **Validate carefully**: Always check `metrics.json` to ensure distilled model performance is acceptable
4. **Choose appropriate base models**:
   - SetFit: Use sentence transformer models (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
   - Model2Vec: Use static embedding models (e.g., `minishlab/potion-base-8M`)
5. **Split data wisely**: Reserve 20-30% for validation (`val_frac=0.2` is a good default)
6. **Iterate**: If distilled performance is poor, try collecting more diverse examples or using a larger base model

> [!TIP]
> Start with a small dataset (50-100 examples) to validate your distillation workflow before scaling up. This helps catch configuration issues early without wasting computational resources.

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

| Task | `task.distill()` | `to_hf_dataset()` |
|------|------------------|-------------------|
| **Classification** | ✅ SetFit, Model2Vec | ✅ |
| **Sentiment Analysis** | ❌ | ✅ |
| **NER** | ❌ | ✅ |
| **PII Masking** | ❌ | ✅ |
| **Information Extraction** | ❌ | ✅ |
| **Summarization** | ❌ | ✅ |
| **Translation** | ❌ | ❌ (NotImplementedError) |
| **Question Answering** | ❌ | ✅ |

> [!IMPORTANT]
> For tasks without `task.distill()` support, use `to_hf_dataset()` to export results, then train with your preferred framework. All tasks except Translation support dataset export.

## Further Reading

- [SetFit Documentation](https://huggingface.co/docs/setfit/)
- [Model2Vec Documentation](https://github.com/MinishLab/model2vec)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- [Task-specific documentation](../tasks/predictive/classification.md) for details on each task's output format
