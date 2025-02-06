# Getting Started with `sieves`

Welcome to the `sieves` docs!

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
pip install "sieves[all]"
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
2. Install all dependencies for development, testing, documentation generation with: `uv pip install --system .[all,test]`.

## Core Concepts

`sieves` is built around five key components:

1. **`Pipeline`**: The main orchestrator that runs your NLP tasks sequentially
2. **`Task`**: Pre-built or custom NLP operations (classification, extraction, etc.)
3. **`Engine`**: Backend implementations that power the tasks (outlines, dspy, langchain, etc.)
4. **`Bridge`**: Connectors between Tasks and Engines
5. **`Doc`**: The fundamental data structure for document processing

## Documentation Structure

Our documentation is organized as follows:

- `/docs/guides/` - Step-by-step tutorials and how-to guides
- `/docs/tasks/` - Detailed documentation for all available tasks
- `/docs/engines/` - Information about supported engines and their configurations
- `/docs/doc.md` - Documentation about the Doc class and its usage
- `/docs/bridge.md` - Understanding the Bridge system
- `/docs/pipeline.md` - Pipeline configuration and advanced usage

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

## Supported Engines

`sieves` supports multiple structured generation libraries:
- [`outlines`](https://github.com/outlines-dev/outlines)
- [`dspy`](https://github.com/stanfordnlp/dspy)
- [`instructor`](https://github.com/instructor-ai/instructor)
- [`langchain`](https://github.com/langchain-ai/langchain)
- [`gliner`](https://github.com/urchade/GLiNER)
- [`transformers`](https://github.com/huggingface/transformers)
- [`ollama`](https://github.com/ollama/ollama)

## Best Practices

1. Start with simple pipelines and gradually add complexity
2. Chunk large documents
3. Leverage built-in tasks before creating custom ones
4. Validate end evaluate task outputs for quality control
5. Save and version your pipeline configurations

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
