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

```python
import outlines
from sieves import Pipeline, tasks, Doc

# Create a document
doc = Doc(text="Special relativity applies to all physical phenomena in the absence of gravity.")

# Choose a model (using a small but capable model)
model = outlines.models.transformers("HuggingFaceTB/SmolLM-135M-Instruct")

# Create and run the pipeline (verbose init)
pipeline = Pipeline([
    tasks.predictive.Classification(
        labels=["science", "politics"],
        model=model,
    )
])

# Print the classification result
for doc in pipeline([doc]):
    print(doc.results)

# Alternatively: succinct chaining with +
# (useful when you have multiple tasks)
# classifier = tasks.predictive.Classification(labels=["science", "politics"], model=model)
# pipeline = classifier  # single-task pipeline
# Note: set additional Pipeline params (e.g., use_cache=False) only via verbose init.
```

## Working with Documents

Documents can be created in several ways:

```python
from sieves import Docs

# From text
doc = Doc(text="Your text here")

# From a file (requires docling)
doc = Doc(uri="path/to/your/file.pdf")

# With metadata
doc = Doc(
    text="Your text here",
    meta={"source": "example", "date": "2025-01-31"}
)
```

## Advanced Example: PDF Processing Pipeline

Here's a more involved example that:

1. Parses a PDF document
2. Chunks it into smaller pieces
3. Performs information extraction on each chunk

```python
import outlines
import chonkie
import tokenizers
import pydantic
from sieves import Pipeline, tasks, Doc

# Create a tokenizer for chunking
tokenizer = tokenizers.Tokenizer.from_pretrained("bert-base-uncased")

# Initialize components
chunker = tasks.preprocessing.Chonkie(
    chunker=chonkie.TokenChunker(tokenizer, chunk_size=512, chunk_overlap=50)
)

# Choose a model for information extraction
model = outlines.models.transformers("HuggingFaceTB/SmolLM-135M-Instruct")


# Define the structure of information you want to extract
class PersonInfo(pydantic.BaseModel):
    name: str
    age: int | None = None
    occupation: str | None = None


# Create an information extraction task
extractor = tasks.predictive.InformationExtraction(
    entity_type=PersonInfo,
    model=model,
)

# Create the pipeline (verbose init)
pipeline = Pipeline([chunker, extractor])

# Alternatively: succinct chaining (+)
# pipeline = chunker + extractor
# Note: to change Pipeline parameters (e.g., use_cache), use the verbose form
#   Pipeline([chunker, extractor], use_cache=False)

# Process a PDF document
doc = Doc(text="Marie Curie died at the age of 66 years.")
results = list(pipeline([doc]))

# Access the extracted information
for result in results:
    print(result.results["InformationExtraction"])
```

## Supported Engines

`sieves` supports multiple libraries for structured generation:

- [`outlines`](https://github.com/outlines-dev/outlines)
- [`dspy`](https://github.com/stanfordnlp/dspy)
- [`instructor`](https://github.com/instructor-ai/instructor)
- [`langchain`](https://github.com/langchain-ai/langchain)
- [`gliner`](https://github.com/urchade/GLiNER)
- [`transformers`](https://github.com/huggingface/transformers)
- [`ollama`](https://github.com/ollama/ollama)

You pass models from these libraries directly to `PredictiveTask`. Optionally, you can include `GenerationSettings` to
override defaults.

### GenerationSettings (optional)
`GenerationSettings` controls execution behavior across engines and is optional. Defaults:
- batch_size: -1 (batch all inputs together)
- strict_mode: False (on parse issues, return None instead of raising)
- init_kwargs/inference_kwargs: None (use engine defaults)
- config_kwargs: None (used by some backends like DSPy)

Only pass it if you need non-defaults, for example:

```python
from sieves.engines.utils import GenerationSettings
classifier = tasks.predictive.Classification(
    labels=["science", "politics"],
    model=model,
    generation_settings=GenerationSettings(batch_size=8, strict_mode=True),
)
```
