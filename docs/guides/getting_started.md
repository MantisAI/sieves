# Getting Started

This guide will help you get started with using `sieves` for zero-shot and few-shot NLP tasks with structured generation.

## Basic Concepts

`sieves` is built around four main concepts:

1. **Documents (`Doc`)**: The basic unit of text that you want to process. A document can be created from text or a URI.
2. **Models + GenerationSettings**: You pass a model from your chosen backend (Outlines, DSPy, LangChain, etc.) and optional `GenerationSettings` (e.g., batch size)
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

## Supported Engines

`sieves` supports multiple libraries for structured generation:

- [`outlines`](https://github.com/outlines-dev/outlines)
- [`dspy`](https://github.com/stanfordnlp/dspy) - also supports Ollama and vLLM integration via `api_base`
- [`langchain`](https://github.com/langchain-ai/langchain)
- [`gliner2`](https://github.com/fastino-ai/GLiNER2)
- [`transformers`](https://github.com/huggingface/transformers)

You pass models from these libraries directly to `PredictiveTask`. Optionally, you can include `GenerationSettings` to
override defaults. Batching is controlled per task via the `batch_size` argument (see below).

### GenerationSettings (optional)
`GenerationSettings` controls engine behavior and is optional. Defaults:
- strict_mode: False (on parse issues, return None instead of raising)
- init_kwargs/inference_kwargs: None (use engine defaults)
- config_kwargs: None (used by some backends like DSPy)
- inference_mode: None (use engine defaults; specifies how the engine queries the model and parses results)

Batching is configured on each task via `batch_size`:
- `batch_size = -1` processes all inputs at once (default)
- `batch_size = N` processes N docs per batch

Example:

```python title="Configuring generation settings and batching"
--8<-- "sieves/tests/docs/test_getting_started.py:generation-settings-config"
```

To specify an inference mode (engine-specific):

```python title="Engine-specific inference mode configuration"
--8<-- "sieves/tests/docs/test_getting_started.py:inference-mode-config"
```
