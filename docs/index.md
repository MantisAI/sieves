# `sieves`

`sieves` is a Python library designed for zero-shot and few-shot NLP tasks that focuses on structured generation,
allowing developers to build production-ready NLP prototypes without requiring training data. It provides a unified
interface that wraps popular NLP tools (like `outlines`, `dspy`, `langchain`, and others) while ensuring structured
outputs and observability.

It bundles common NLP utilities, document parsing, and text chunking capabilities together with ready-to-use tasks like
classification and information extraction, all organized in an observable pipeline architecture. It's particularly
valuable for rapid prototyping scenarios where structured output is needed but training data is scarce.


## Quick Installation

You can install `sieves` with different options depending on your needs

Core package with minimal dependencies:
```bash
pip install sieves
```
Note that `sieves` relies on the functionality of a lot of other libraries to work properly. The minimal setup allows
you to manually install only the dependencies you need to keep the disk footprint small, but keep in mind you won't be
able to use any of the pre-built tasks with this setup.

All  dependencies for every feature, including all supported engines and utilities:
```bash
pip install "sieves[engines,distill]"
```

### Specific Features

All document processing utilities (PDF parsing, chunking, etc.):
```bash
pip install "sieves[utils]"
```

All supported engines:
```bash
pip install "sieves[engines]"
```

### Development Setup

1. Set up [`uv`](https://github.com/astral-sh/uv).
2. Install all dependencies for development, testing, documentation generation with: `uv pip install --system .[engines,distill,test]`.

## Core Concepts

`sieves` is built around five key components:

1. **`Pipeline`**: The main orchestrator that runs your NLP tasks sequentially (define with `Pipeline([...])` or chain with `+`)
2. **`Task`**: Pre-built or custom NLP operations (classification, extraction, etc.)
3. **`Engine`**: Backend implementations that power the tasks (outlines, dspy, langchain, etc.)
4. **`Bridge`**: Connectors between Tasks and Engines
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

## Getting Help

- Check our [GitHub Issues](https://github.com/mantisai/sieves/issues) for common problems
- Review the documentation in the `/docs/guides/` directory
- Join our community discussions (link to be added)

## Next Steps

- Dive into our guides, starting with the [Getting Started Guide](guides/getting_started.md)
- Check out example pipelines in our repository
- Learn about custom task creation
- Understand different engine configurations

Consult the API reference for each component you're working with if you have specific question. They contain detailed
information about parameters, configurations, and best practices.
