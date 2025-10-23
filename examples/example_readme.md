---
viewer: false
tags: [sieves-script, classification, outlines, structured-generation, gpu-recommended]
---

# Dataset Classification with `sieves`

GPU-accelerated text classification for Hugging Face datasets with guaranteed valid outputs through structured
generation with [Sieves](https://github.com/MantisAI/sieves/), [Outlines](https://github.com/dottxt-ai/outlines) and
Hugging Face zero-shot pipelines.

This is a modified version of https://huggingface.co/datasets/uv-scripts/classification.

## 🚀 Quick Start

```bash
# Classify IMDB reviews
uv run examples/classify-dataset.py \
  --input-dataset stanfordnlp/imdb \
  --column text \
  --labels "positive,negative" \
  --model HuggingFaceTB/SmolLM-360M-Instruct \
  --output-dataset user/imdb-classified
```

That's it! No installation, no setup - just `uv run`.

## 📋 Requirements

- **GPU Recommended**: Uses GPU-accelerated inference (CPU fallback available but slow)
- Python 3.12+
- UV (will handle all dependencies automatically)

**Python Package Dependencies** (automatically installed via UV):
- `sieves` with engines support (>= 0.17.4)
- `typer` (>= 0.12)
- `datasets`
- `huggingface-hub`

## 🎯 Features

- **Guaranteed valid outputs** using structured generation with Outlines guided decoding
- **Zero-shot classification** without training data required
- **GPU-optimized** for maximum throughput and efficiency
- **Multi-label support** for documents with multiple applicable labels
- **Flexible model selection** - works with any instruction-tuned transformer model
- **Robust text handling** with preprocessing and validation
- **Automatic progress tracking** and detailed statistics
- **Direct Hub integration** - read and write datasets seamlessly
- **Label descriptions** support for providing context to improve accuracy
- **Optimized batching** with Sieves' automatic batch processing
- **Multiple guided backends** - supports `outlines` to handle any general language model on Hugging Face, and fast Hugging Face zero-shot classification pipelines

## 💻 Usage

### Basic Classification

```bash
uv run examples/classify-dataset.py \
  --input-dataset <dataset-id> \
  --column <text-column> \
  --labels <comma-separated-labels> \
  --model <model-id> \
  --output-dataset <output-id>
```

### Arguments

**Required:**

- `--input-dataset`: Hugging Face dataset ID (e.g., `stanfordnlp/imdb`, `user/my-dataset`)
- `--column`: Name of the text column to classify
- `--labels`: Comma-separated classification labels (e.g., `"spam,ham"`)
- `--model`: Model to use (e.g., `HuggingFaceTB/SmolLM-360M-Instruct`)
- `--output-dataset`: Where to save the classified dataset

**Optional:**

- `--label-descriptions`: Provide descriptions for each label to improve classification accuracy
- `--multi-label`: Enable multi-label classification mode (creates multi-hot encoded labels)
- `--split`: Dataset split to process (default: `train`)
- `--max-samples`: Limit samples for testing
- `--shuffle`: Shuffle dataset before selecting samples (useful for random sampling)
- `--shuffle-seed`: Random seed for shuffling
- `--batch-size`: Batch size for inference (default: 64)
- `--max-tokens`: Maximum tokens to generate per sample (default: 200)
- `--hf-token`: Hugging Face token (or use `HF_TOKEN` env var)

### Label Descriptions

Provide context for your labels to improve classification accuracy:

```bash
uv run examples/classify-dataset.py \
  --input-dataset user/support-tickets \
  --column content \
  --labels "bug,feature,question,other" \
  --label-descriptions "bug:something is broken,feature:request for new functionality,question:asking for help,other:anything else" \
  --model HuggingFaceTB/SmolLM-360M-Instruct \
  --output-dataset user/tickets-classified
```

The model uses these descriptions to better understand what each label represents, leading to more accurate classifications.

### Multi-Label Classification

Enable multi-label mode for documents that can have multiple applicable labels:

```bash
uv run examples/classify-dataset.py \
  --input-dataset ag_news \
  --column text \
  --labels "world,sports,business,science" \
  --multi-label \
  --model HuggingFaceTB/SmolLM-360M-Instruct \
  --output-dataset user/ag-news-multilabel
```

## 📊 Examples

### Sentiment Analysis

```bash
uv run examples/classify-dataset.py \
  --input-dataset stanfordnlp/imdb \
  --column text \
  --labels "positive,ambivalent,negative" \
  --model HuggingFaceTB/SmolLM-360M-Instruct \
  --output-dataset user/imdb-sentiment
```

### Support Ticket Classification

```bash
uv run examples/classify-dataset.py \
  --input-dataset user/support-tickets \
  --column content \
  --labels "bug,feature_request,question,other" \
  --label-descriptions "bug:code or product not working as expected,feature_request:asking for new functionality,question:seeking help or clarification,other:general comments or feedback" \
  --model HuggingFaceTB/SmolLM-360M-Instruct \
  --output-dataset user/tickets-classified
```

### News Categorization

```bash
uv run examples/classify-dataset.py \
  --input-dataset ag_news \
  --column text \
  --labels "world,sports,business,tech" \
  --model HuggingFaceTB/SmolLM-1.7B-Instruct \
  --output-dataset user/ag-news-categorized
```

### Multi-Label News Classification

```bash
uv run examples/classify-dataset.py \
  --input-dataset ag_news \
  --column text \
  --labels "world,sports,business,tech" \
  --multi-label \
  --label-descriptions "world:global and international events,sports:sports and athletics,business:business and finance,tech:technology and innovation" \
  --model HuggingFaceTB/SmolLM-1.7B-Instruct \
  --output-dataset user/ag-news-multilabel
```

This combines label descriptions with multi-label mode for comprehensive categorization of news articles.

### ArXiv ML Research Classification

Classify academic papers into machine learning research areas:

```bash
# Fast classification with random sampling
uv run examples/classify-dataset.py \
  --input-dataset librarian-bots/arxiv-metadata-snapshot \
  --column abstract \
  --labels "llm,computer_vision,reinforcement_learning,optimization,theory,other" \
  --label-descriptions "llm:language models and NLP,computer_vision:image and video processing,reinforcement_learning:RL and decision making,optimization:training and efficiency,theory:theoretical ML foundations,other:other ML topics" \
  --model HuggingFaceTB/SmolLM-360M-Instruct \
  --output-dataset user/arxiv-ml-classified \
  --split "train" \
  --max-samples 100 \
  --shuffle

# Multi-label for nuanced classification
uv run examples/classify-dataset.py \
  --input-dataset librarian-bots/arxiv-metadata-snapshot \
  --column abstract \
  --labels "multimodal,agents,reasoning,safety,efficiency" \
  --label-descriptions "multimodal:vision-language and cross-modal models,agents:autonomous agents and tool use,reasoning:reasoning and planning systems,safety:alignment and safety research,efficiency:model optimization and deployment" \
  --multi-label \
  --model HuggingFaceTB/SmolLM-360M-Instruct \
  --output-dataset user/arxiv-frontier-research \
  --split "train[:1000]" \
  --max-samples 50
```

Multi-label mode is particularly valuable for academic abstracts where papers often span multiple topics and require careful analysis to determine all relevant research areas.

## 🚀 Running Locally vs Cloud

This script is optimized to run locally on GPU-equipped machines:

```bash
# Local execution with your GPU
uv run examples/classify-dataset.py \
  --input-dataset stanfordnlp/imdb \
  --column text \
  --labels "positive,negative" \
  --model HuggingFaceTB/SmolLM-360M-Instruct \
  --output-dataset user/imdb-classified
```

For cloud deployment, you can use Hugging Face Spaces or other GPU services by adapting the command to your environment.

## 🔧 Advanced Usage

### Random Sampling

When working with ordered datasets, use `--shuffle` with `--max-samples` to get a representative sample:

```bash
# Get 50 random reviews instead of the first 50
uv run examples/classify-dataset.py \
  --input-dataset stanfordnlp/imdb \
  --column text \
  --labels "positive,negative" \
  --model HuggingFaceTB/SmolLM-360M-Instruct \
  --output-dataset user/imdb-sample \
  --max-samples 50 \
  --shuffle \
  --shuffle-seed 123  # For reproducibility
```


### Using Different Models

By default, this script works with any instruction-tuned model. Here are some recommended options:

```bash
# Lightweight model for fast classification
uv run examples/classify-dataset.py \
  --input-dataset user/my-dataset \
  --column text \
  --labels "A,B,C" \
  --model HuggingFaceTB/SmolLM-360M-Instruct \
  --output-dataset user/classified

# Larger model for complex classification
uv run examples/classify-dataset.py \
  --input-dataset user/legal-docs \
  --column text \
  --labels "contract,patent,brief,memo,other" \
  --model HuggingFaceTB/SmolLM3-3B-Instruct \
  --output-dataset user/legal-classified

# Specialized zero-shot classifier
uv run examples/classify-dataset.py \
  --input-dataset user/my-dataset \
  --column text \
  --labels "A,B,C" \
  --model MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
  --output-dataset user/classified
```

### Large Datasets

Configure `--batch-size` for more effective batch processing with large datasets:

```bash
uv run examples/classify-dataset.py \
  --input-dataset user/huge-dataset \
  --column text \
  --labels "A,B,C" \
  --model HuggingFaceTB/SmolLM-360M-Instruct \
  --output-dataset user/huge-classified \
  --batch-size 128
```


## 🤝 How It Works

1. **Sieves**: Provides a zero-shot task pipeline system for structured NLP workflows
2. **Outlines**: Provides guided decoding to guarantee valid label outputs
3. **UV**: Handles all dependencies automatically

The script loads your dataset, preprocesses texts, classifies each one with guaranteed valid outputs using Sieves'
`Classification` task, then saves the results as a new column in the output dataset.

## 🐛 Troubleshooting

### GPU Not Available

This script works best with a GPU but can run on CPU (much slower). To use GPU:

- Run on a machine with NVIDIA GPU
- Use cloud GPU instances (AWS, GCP, Azure, etc.)
- Use Hugging Face Spaces with GPU

### Out of Memory

- Use a smaller model (e.g., SmolLM-360M instead of 3B)
- Reduce `--batch-size` (try 32, 16, or 8)
- Reduce `--max-tokens` for shorter generations

### Invalid/Skipped Texts

- Texts shorter than 3 characters are skipped
- Empty or None values are marked as invalid
- Very long texts are truncated to 4000 characters

### Classification Quality

- With Outlines guided decoding, outputs are guaranteed to be valid labels
- For better results, use clear and distinct label names
- Try `--label-descriptions` to provide context
- Use a larger model for nuanced tasks
- In multi-label mode, adjust the confidence threshold (defaults to 0.5)

### Authentication Issues

If you see authentication errors:

- Run `huggingface-cli login` to cache your token
- Or set `export HF_TOKEN=your_token_here`
- Verify your token has read/write permissions on the Hub

## 🔬 Advanced Workflows

### Full Pipeline Workflow

Start with small tests, then run on the full dataset:

```bash
# Step 1: Test with small sample
uv run examples/classify-dataset.py \
  --input-dataset your-dataset \
  --column text \
  --labels "label1,label2,label3" \
  --model HuggingFaceTB/SmolLM-360M-Instruct \
  --output-dataset user/test-classification \
  --max-samples 100

# Step 2: If results look good, run on full dataset
uv run examples/classify-dataset.py \
  --input-dataset your-dataset \
  --column text \
  --labels "label1,label2,label3" \
  --label-descriptions "label1:description,label2:description,label3:description" \
  --model HuggingFaceTB/SmolLM-360M-Instruct \
  --output-dataset user/final-classification \
  --batch-size 64
```

## 📝 License

This example is provided as part of the [Sieves](https://github.com/MantisAI/sieves/) project.
