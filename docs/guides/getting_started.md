# Getting Started

This guide will help you get started with using `sieves` for zero-shot and few-shot NLP tasks with structured generation.

## Basic Concepts

`sieves` is built around four main concepts:

1. **Documents (`Doc`)**: The basic unit of text that you want to process. A document can be created from text or a URI.
2. **Models + ModelSettings**: You pass a model from your chosen backend (Outlines, DSPy, LangChain, etc.) and optional `ModelSettings` (e.g., batch size)
3. **Tasks**: NLP operations you want to perform on your documents (classification, information extraction, etc.)
4. **Pipeline**: A sequence of tasks that process your documents

## Quick Start Example

Here's a simple example that performs text classification:

```python title="Basic text classification"
--8<-- "sieves/tests/docs/test_getting_started.py:basic-classification"
```

### Using Label Descriptions

You can improve classification accuracy by providing descriptions for each label. This is especially helpful when label names alone might be ambiguous:

```python title="Classification with label descriptions"
--8<-- "sieves/tests/docs/test_getting_started.py:label-descriptions"
```

## Working with Documents

Documents can be created in several ways:

```python title="Creating documents from text"
--8<-- "sieves/tests/docs/test_getting_started.py:doc-from-text"
```

```python title="Creating documents from a file (requires ingestion extra)"
--8<-- "sieves/tests/docs/test_getting_started.py:doc-from-uri"
```

```python title="Creating documents with metadata"
--8<-- "sieves/tests/docs/test_getting_started.py:doc-with-metadata"
```

Note: File-based ingestion (Docling/Marker/...) is optional and not installed by default. To enable it, install the ingestion extra or the specific libraries you need:

```bash
pip install "sieves[ingestion]"
```

## Advanced Example: PDF Processing Pipeline

Here's a more involved example that:

1. Parses a PDF document
2. Chunks it into smaller pieces
3. Performs information extraction on each chunk

```python title="Advanced pipeline with chunking and extraction"
--8<-- "sieves/tests/docs/test_getting_started.py:advanced-pipeline"
```

## Supported Models

`sieves` supports multiple frameworks for interacting with zero-shot language models - see
https://sieves.ai/guides/models/ for an overview.

You pass supported models directly to `PredictiveTask`. Optionally, you can include `ModelSettings` to
influence model initialization and runtime behavior.

Batching is configured on each task via `batch_size`:
- `batch_size = -1` processes all inputs at once (default)
- `batch_size = N` processes N docs per batch

### ModelSettings (optional)
`ModelSettings` controls details of how the model's structured generation will be run. It allows to configure:

- `strict`: Whether to raise encountered errors, or assign placeholder values for documents that failed to process.
- `init_kwargs`: Model-specific arguments that will be passed to the model's structured generation abstraction at its initialization.
- `inference_kwargs`: Model-specific arguments that will be passed to the model's structured generation abstraction during inference.
- `config_kwargs`: Model-specific arguments that will be applied to the model after task initialization.
- `inference_mode`: Model-specific modes for structured generation. Don't change this unless you _exactly_ know what you're doing.

Example:

```python title="Configuring model settings and batching"
--8<-- "sieves/tests/docs/test_getting_started.py:generation-settings-config"
```

To specify an inference mode (model type-specific):

```python title="ModelWrapper-specific inference mode configuration"
--8<-- "sieves/tests/docs/test_getting_started.py:inference-mode-config"
```
