# `sieves`

`sieves` is a library for zero-shot document AI with structured generation. It facilitates the rapid prototyping of
document AI pipelines with validated output. No training required.

It bundles common NLP utilities, document parsing, and text chunking capabilities together with ready-to-use tasks like
classification and information extraction, all organized in an observable pipeline architecture. It's particularly
valuable for rapid prototyping scenarios where structured output is needed but training data is scarce.

## Quick Installation

You can install `sieves` with different options depending on your needs

Core package:
```bash
pip install sieves
```

Ingestion libraries (document parsing such as `docling`) are optional. Install them manually or use the ingestion extra:

```bash
pip install "sieves[ingestion]"
```

The minimal setup lets you add only what you need to keep the footprint small.

All optional dependencies for every feature, including distillation and ingestion:
```bash
pip install "sieves[distill,ingestion]"
```

### Specific Features

Document ingestion/parsing libraries (PDF/DOCX parsing, etc.):
```bash
pip install "sieves[ingestion]"
```

Distillation utilities for model fine-tuning:
```bash
pip install "sieves[distill]"
```

### Development Setup

1. Set up [`uv`](https://github.com/astral-sh/uv).
2. Install all dependencies for development, testing, documentation generation with: `uv pip install --system .[distill,ingestion,test]`.

## Core Concepts

`sieves` is built around five key components:

1. **`Pipeline`**: The main orchestrator that runs your NLP tasks sequentially (define with `Pipeline([...])` or chain with `+`)
2. **`Task`**: Pre-built or custom NLP operations (classification, extraction, etc.)
3. **`ModelWrapper`**: Backend implementations that power the tasks (outlines, dspy, langchain, etc.)
4. **`Bridge`**: Connectors between Tasks and Model wrappers
5. **`Doc`**: The fundamental data structure for document processing

## Essential Links

- [GitHub Repository](https://github.com/mantisai/sieves)
- [PyPI Package](https://pypi.org/project/sieves/)
- [Issue Tracker](https://github.com/mantisai/sieves/issues)

## Guides

We've prepared several guides to help you get up to speed quickly:

- [Getting Started](guides/getting_started.md) - Start here! Learn the basic concepts and create your first pipeline.
- [Document Preprocessing](guides/preprocessing.md) - Master document parsing, chunking, and text standardization.
- [Creating Custom Tasks](guides/custom_tasks.md) - Learn to create your own tasks when the built-in ones aren't enough.
- [Saving and Loading Pipelines](guides/serialization.md) - Version and share your pipeline configurations.
- [Task Optimization](guides/optimization.md) - Improve task performance by optimizing prompts and few-shot examples.
- [Task Distillation](guides/distillation.md) - Fine-tune smaller, specialized models using zero-shot task results.

## Getting Help

- Check our [GitHub Issues](https://github.com/mantisai/sieves/issues) for common problems
- Review the documentation in the `/docs/guides/` directory
- Join our community discussions (link to be added)

## Next Steps

- Dive into our guides, starting with the [Getting Started Guide](guides/getting_started.md)
- Check out example pipelines in our repository
- Learn about custom task creation
- Understand different model configurations

Consult the API reference for each component you're working with if you have specific question. They contain detailed
information about parameters, configurations, and best practices.
