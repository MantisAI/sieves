<img src="https://raw.githubusercontent.com/mantisai/sieves/main/docs/assets/sieve.png" width="230" align="left" style="margin-right:60px" />
<img src="https://raw.githubusercontent.com/mantisai/sieves/main/docs/assets/sieves_sieve_style.png" width="350" align="left" style="margin-right:60px" />

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/mantisai/sieves/test.yml)](https://github.com/mantisai/sieves/actions/workflows/test.yml)
![GitHub top language](https://img.shields.io/github/languages/top/mantisai/sieves)
[![PyPI - Version](https://img.shields.io/pypi/v/sieves)]((https://pypi.org/project/sieves/))
![PyPI - Status](https://img.shields.io/pypi/status/sieves)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)
[![codecov](https://codecov.io/gh/mantisai/sieves/branch/main/graph/badge.svg)](https://codecov.io/gh/mantisai/sieves)

## Zero-shot document processing made easy.

`sieves` is a library for zero- and few-shot NLP tasks with structured generation. Build production-ready NLP prototypes quickly, with guaranteed output formats and no training required.

Read our documentation [here](https://sieves.ai). An automatically generated version (courtesy of Devin via [DeepWiki](https://deepwiki.com/)) is available [here](https://deepwiki.com/MantisAI/sieves).

Install `sieves` with `pip install sieves`. If you want to install all optional dependencies right away, install with
`pip install sieves[engines,distill]`. You can also choose to install individual dependencies as you see fit.

### Why `sieves`?

Even in the era of generative AI, structured outputs and observability remain crucial.

Many real-world scenarios require rapid prototyping with minimal data. Generative language models excel here, but
producing clean, structured output can be challenging. Various tools address this need for structured/guided language
model output, including [`outlines`](https://github.com/dottxt-ai/outlines), [`dspy`](https://github.com/stanfordnlp/dspy),
[`ollama`](https://github.com/ollama/ollama), and others. Each has different design patterns, pros and cons. `sieves` wraps these tools and provides
a unified interface for input, processing, and output.

Developing NLP prototypes often involves repetitive steps: parsing and chunking documents, exporting results for
model fine-tuning, and experimenting with different prompting techniques. All these needs are addressed by existing
libraries in the NLP ecosystem address (e.g. [`docling`](https://github.com/DS4SD/docling) for file parsing, or [`datasets`](https://github.com/huggingface/datasets) for transforming
data into a unified format for model training).

`sieves`  **simplifies NLP prototyping** by bundling these capabilities into a single library, allowing you to quickly
build modern NLP applications. It provides:
- Zero- and few-shot model support for immediate inference
- A bundle of utilities addressing common requirements in NLP applications
- A unified interface for structured generation across multiple libraries
- Built-in tasks for common NLP operations
- Easy extendability
- A document-based pipeline architecture for easy observability and debugging
- Caching - pipelines cache processed documents to prevent costly redundant model calls

`sieves` draws a lot of inspiration from [`spaCy`](https://spacy.io/) and particularly [`spacy-llm`](https://github.com/explosion/spacy-llm).

---

### Features

- :dart: **Zero Training Required:** Immediate inference using zero-/few-shot models
- :robot: **Unified Generation Interface:** Seamlessly use multiple libraries
  - [`dspy`](https://github.com/stanfordnlp/dspy)
  - [`gliner`](https://github.com/urchade/GLiNER)
  - [`instructor`](https://github.com/instructor-ai/instructor)
  - [`langchain`](https://github.com/langchain-ai/langchain)
  - [`ollama`](https://github.com/ollama/ollama)
  - [`outlines`](https://github.com/dottxt-ai/outlines)
  - [`transformer`](https://github.com/huggingface/transformers)
- :arrow_forward: **Observable Pipelines:** Easy debugging and monitoring
- :hammer_and_wrench: **Integrated Tools:**
  - Document parsing: [`docling`](https://github.com/DS4SD/docling), [`unstructured`](https://github.com/Unstructured-IO/unstructured/), [`marker`](https://github.com/VikParuchuri/marker)
  - Text chunking: [`chonkie`](https://github.com/chonkie-ai/chonkie)
- :label: **Ready-to-Use Tasks:**
  - Multi-label classification
  - Information extraction
  - Summarization
  - Translation
  - Multi-question answering
  - Aspect-based sentiment analysis
  - PII (personally identifiable information) anonymization
  - Named entity recognition
  - Coming soon: entity linking, knowledge graph creation, ...
- :floppy_disk: **Persistence:** Save and load pipelines with configurations
- :teacher: **Distillation:** Distill local, specialized models using your zero-shot model results automatically.
  Export your results as HuggingFace [`Dataset`](https://github.com/huggingface/datasets) if you want to run your own training routine.
- :recycle: **Caching** to avoid unnecessary model calls

---

### Getting Started

Here's a simple classification example using [`outlines`](https://github.com/dottxt-ai/outlines):
```python
from sieves import Pipeline, Engine, tasks, Doc

# 1. Define documents by text or URI.
docs = [Doc(text="Special relativity applies to all physical phenomena in the absence of gravity.")]

# 2. Create pipeline with tasks (verbose init).
pipe = Pipeline(
    # 3. Add classification task to pipeline.
    # By default Engine uses Outlines with HuggingFaceTB/SmolLM-360M-Instruct. This is a pretty small model, you
    # might want to consider upgrading to a different model for better results.
    tasks.Classification(labels=["science", "politics"], engine=Engine())
)

# 4. Run pipe and output results.
for doc in pipe(docs):
  print(doc.results)

# Tip: Pipelines can also be composed succinctly via chaining (+).
# For multi-step pipelines, you can write:
#   pipe = tasks.Ingestion(export_format="markdown") + tasks.Chunking(chunker) + tasks.Classification(labels=[...], engine=Engine())
# Note: additional Pipeline parameters (e.g., use_cache=False) are only available via the verbose init,
# e.g., Pipeline([t1, t2], use_cache=False).
```

<details>
  <summary><b>Advanced Example</b></summary>

This example demonstrates PDF parsing, text chunking, and classification:
```python
import pickle

import gliner.multitask
import chonkie
import tokenizers
import docling.document_converter

from sieves import Pipeline, Engine, tasks, Doc

# 1. Define documents by text or URI.
docs = [Doc(uri="https://arxiv.org/pdf/2408.09869")]

# 2. Create engine responsible for generating structured output.
model_name = 'knowledgator/gliner-multitask-v1.0'
engine = Engine(model=gliner.GLiNER.from_pretrained(model_name))

# 3. Create chunker object.
chunker = chonkie.TokenChunker(tokenizers.Tokenizer.from_pretrained(model_name))

# 3. Create pipeline with tasks.
pipe = Pipeline(
    [
        # 4. Add document parsing task.
        tasks.Ingestion(export_format="markdown"),
        # 5. Add chunking task to ensure we don't exceed our model's context window.
        tasks.Chunking(chunker),
        # 6. Add classification task to pipeline.
        tasks.Classification(task_id="classifier", labels=["science", "politics"], engine=engine),
    ]
)
# Alternatively you can also construct a pipeline by using the + operators:
# pipe = tasks.Ingestion(export_format="markdown") + tasks.Chunking(chunker) + tasks.Classification(task_id="classifier", labels=["science", "politics"], engine=engine)

# 7. Run pipe and output results.
docs = list(pipe(docs))
for doc in docs:
    print(doc.results["classifier"])

# 8. Serialize pipeline and docs.
pipe.dump("pipeline.yml")
with open("docs.pkl", "wb") as f:
    pickle.dump(docs, f)

# 9. Load pipeline and docs from disk. Note: we don't serialize complex third-party objects, so you'll have
#    to pass those in at load time.
loaded_pipe = Pipeline.load(
    "pipeline.yml",
    (
        {"converter": docling.document_converter.DocumentConverter(), "export_format": "markdown"},
        {"chunker": chunker},
        {"engine": {"model": engine.model}},
    ),
)
with open("docs.pkl", "rb") as f:
    loaded_docs = pickle.load(f)
```
</details>

---

### Core Concepts

`sieves` is built on five key abstractions.

#### **`Pipeline`**
Orchestrates task execution with features for.
- Task configuration and sequencing
- Pipeline execution
- Configuration management and serialization

#### **`Doc`**
Represents a document in the pipeline.
- Contains text content and metadata
- Tracks document URI and processing results
- Passes information between pipeline tasks

#### **`Task`**
Encapsulates a single processing step in a pipeline.
- Defines input arguments
- Wraps and initializes `Bridge` instances handling task-engine-specific logic
- Implements task-specific dataset export

#### **`Engine`**
Provides a unified interface to structured generation libraries.
- Manages model interactions
- Handles prompt execution
- Standardizes output formats

#### **`Bridge`**
Connects `Task` with `Engine`.
- Implements engine-specific prompt templates
- Manages output type specifications
- Ensures compatibility between tasks and engine

---

## Frequently Asked Questions

<details>
  <summary><b>Show FAQs</b></summary>

### Why "sieves"?

`sieves` was originally motivated by the want to use generative models for structured information extraction. Coming
from this angle, there are two ways to explain why we settled on this name (pick the one you like better):
- An analogy to [gold panning](https://en.wikipedia.org/wiki/Gold_panning): run your raw data through a sieve to obtain structured, refined "gold."
- An acronym - "sieves" can be read as "Structured Information Extraction and VErification System" (but that's a mouthful).

### Why not just prompt an LLM directly?

Asked differently: what are the benefits of using `sieves` over directly interacting with an LLM?
- Validated, structured data output - also for LLMs that don't offer structured outputs natively.  Zero-/few-shot language models can be finicky without guardrails or parsing.
- A step-by-step pipeline, making it easier to debug and track each stage.
- The flexibility to switch between different models and ways to ensure structured and validated output.
- A bunch of useful utilities for pre- and post-processing you might need.
- An array of useful tasks you can right of the bat without having to roll your own.

### Why use `sieves` and not a structured generation library, like `outlines`, directly?

Which library makes the most sense to you depends strongly on your use-case. `outlines` provides structured generation
abilities, but not the pipeline system, utilities and pre-built tasks that `sieves` has to offer (and of course not the
flexibility to switch between different structured generation libraries). Then again, maybe you don't need all that -
in which case we recommend using `outlines` (or any other structured generation libray) directly.

Similarly, maybe you already have an existing tech stack in your project that uses exclusively `ollama`, `langchain`, or
`dspy`? All of these libraries (and more) are supported by `sieves` - but they are not _just_ structured generation
libraries, they come with a plethora of features that are out of scope for `sieves`. If your application deeply
integrates with a framework like LangChain or DSPy, it may be reasonable to stick to those libraries directly.

As many things in engineering, this is a trade-off. The way we see it: the less tightly coupled your existing
application is with a particular language model framework, the more mileage you'll get out of `sieves`. This means that
it's ideal for prototyping (there's no reason you can't use it in production too, of course).

</details>

---

> Source for `sieves` icon:
> <a href="https://www.flaticon.com/free-icons/sieve" title="sieve icons">Sieve icons created by Freepik - Flaticon</a>.
