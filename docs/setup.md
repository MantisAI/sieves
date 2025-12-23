# Installation


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

## Installation with Optional Extras

Document ingestion/parsing libraries (PDF/DOCX parsing, etc.):
```bash
pip install "sieves[ingestion]"
```

Distillation utilities for model fine-tuning:
```bash
pip install "sieves[distill]"
```

## Development Setup

1. Set up [`uv`](https://github.com/astral-sh/uv).
2. Install all dependencies for development, testing, documentation generation with: `uv pip install --system .[distill,ingestion,test]`.
