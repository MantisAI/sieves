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
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17633730.svg)](https://doi.org/10.5281/zenodo.17633730)


## Zero-shot document processing made easy.

`sieves` is a library for zero-shot document AI with structured generation.
It supports you to build document AI pipelines quickly, with validated output formats. No training required.

Read our documentation at [sieves.ai](https://sieves.ai). An automatically generated version (courtesy of Devin via [DeepWiki](https://deepwiki.com/)) is available [here](https://deepwiki.com/MantisAI/sieves).

> [!WARNING]
> `sieves` is in active development and currently in beta. Be advised that the API might change in between minor version
> updates.

## Quick Start

**1. Install**
```bash
pip install sieves
```

**2. Run this example** - Text classification with local models:

```python
import outlines
import transformers
from sieves import Pipeline, tasks, Doc

# Create model and pipeline
model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
model = outlines.models.from_transformers(
    transformers.AutoModelForCausalLM.from_pretrained(model_name),
    transformers.AutoTokenizer.from_pretrained(model_name)
)

pipeline = Pipeline(
    tasks.Classification(
        labels=["technology", "sports", "politics"],
        model=model
    )
)

# Process text
doc = Doc(text="The new smartphone features advanced AI capabilities.")
results = list(pipeline([doc]))
print(results[0].results)  # "technology"
```

**3. Explore the docs**

[Read the guides](https://sieves.ai/guides/getting_started) • [Browse examples](https://sieves.ai/examples)

<details>
  <summary><b>Advanced Example: Information Extraction from PDFs</b></summary>

This example shows how to extract structured information from a scientific paper as PDF using
- a remote model via DSPy and OpenRouter
- chunking
- the `InformationExtraction` task

We're using this setup to extract mathematical equations from the paper.
```python
    import dspy
    import os
    from sieves import tasks, Doc
    import pydantic
    import chonkie

    # Define schema for extraction.
    class Equation(pydantic.BaseModel, frozen=True):
        id: str = pydantic.Field(description="ID/index of equation in paper.")
        equation: str = pydantic.Field(description="Equation as shown in paper.")

    # Create model instance using OpenRouter.
    model = dspy.LM(
        "openrouter/google/gemini-2.5-flash-lite-preview-09-2025",
        api_base="https://openrouter.ai/api/v1/",
        api_key=os.environ["OPENROUTER_API_KEY"]
    )

    # Create pipeline with PDF ingestion, chunking, and extraction.
    pipeline = (
        tasks.Ingestion(export_format="markdown") +
        tasks.Chunking(chonkie.TokenChunker(tokenizers.Tokenizer.from_pretrained("gpt2"))) +
        tasks.InformationExtraction(entity_type=Equation, model=model)
    )

    # Process a paper with equations as PDF.
    pdf_path = "https://arxiv.org/pdf/1204.0162"
    doc = Doc(uri=pdf_path)
    results = list(pipeline([doc]))

    # Access extracted entities.
    if results[0].results.get("InformationExtraction"):
        for equation in results[0].results["InformationExtraction"]:
            print(equation)
```
The output will look similar to this:
```
id='(1)' equation="the observer measures not the linear but angular ... both cars are near the stop sign."
id='(3)' equation='\\omega(t) = \\frac{r_0 v(t)}{r_0^2 + x(t)^2}'
id='(4)' equation='\\tan \\alpha(t) = \\frac{x(t)}{r_0}'
id='(5)' equation='x(t) = \\frac{a_0 t^2}{2}'
id='(6)' equation="\\frac{d}{dt} f(t) = f'(t)"
id='(7)' equation='\\omega(t) = \\frac{a_0 t}{r_0} \\left( 1 + \\frac{a_0^2 t^4}{4 r_0^2} \\right)^{-1}'
id='(8)' equation='x(t) = x_0 + v_0 t + \\frac{1}{2} a t^2'
```

**Requirements**: Install PDF parsing support:
```bash
pip install "sieves[ingestion]"
```

See [Ingestion Guide](https://sieves.ai/guides/ingestion) for more PDF parsing options.

</details>

---

### Key Features

**Zero-shot NLP, ready to use**
- 🎯 No training required - immediate inference with zero-shot models (LLMs and local models)
- 📋 Built-in tasks: classification, extraction, NER, summarization, sentiment analysis, PII masking, QA
- 🔄 Unified interface for DSPy, Outlines, LangChain, GLiNER2, Transformers

**Production-ready pipelines**
- 🔍 Observable execution with conditional task logic
- 💾 Caching to avoid redundant model calls
- 📦 Pipeline serialization and configuration management

**Full NLP workflow**
- 📄 Document parsing: Docling, Marker (optional)
- ✂️ Text chunking: Chonkie integration
- 🚀 Prompt optimization: DSPy MIPROv2
- 👨‍🏫 Model distillation: SetFit, Model2Vec

## Installation

**Requirements**: Python 3.12 (exact version)

```bash
pip install sieves
```

**Optional extras:**
```bash
pip install "sieves[ingestion]"  # PDF/DOCX parsing (docling, marker)
pip install "sieves[distill]"     # Model distillation (setfit, model2vec)
```

> ⚠️ **Important**: sieves requires Python 3.12.x (not 3.13+). Some dependencies like `pyarrow`
> don't have prebuilt wheels for Python 3.13+ yet, which would require manual compilation.
> Support for newer Python libraries is on the roadmap.

## Why `sieves`?

**The challenge**: Building NLP prototypes with LLMs means juggling multiple tools - one for
structured output, another for parsing, one for chunking, another for optimization. There are many options for structured output,
each with its own pros and cons - and very different APIs.

**The solution**: `sieves` provides a unified pipeline for the entire workflow, from document
ingestion to model distillation, with validated structured outputs across multiple backends.

**Best for:**

- ✅ Use case: document AI/processing
- ✅ Rapid prototyping with zero training
- ✅ Switching between language model backends without rewriting code
- ✅ Building document AI pipelines with observability

**Not for:**

- ❌ Use case: chat bot, RAG
- ❌ Applications deeply coupled to LangChain/DSPy ecosystems
- ❌ Simple one-off LLM calls without pipeline needs

Inspired by [spaCy](https://spacy.io/) and [spacy-llm](https://github.com/explosion/spacy-llm).

## How does sieves compare?

| Feature                    | sieves         | LangChain    | DSPy           | Outlines       | Transformers |
|----------------------------|----------------|--------------|----------------|----------------|--------------|
| **Multi-backend support**  | ✅ All          | ❌ Own only   | ❌ Own only     | ❌ Own only     | ❌ Own only   |
| **Document parsing**       | ✅ Built-in     | ✅ Via tools  | ❌ No           | ❌ No           | ❌ No         |
| **Structured output**      | ✅ Unified      | ✅ Yes        | ✅ Yes          | ✅ Core feature | ⚠️ Limited   |
| **Prompt optimization**    | ✅ DSPy wrapper | ❌ No         | ✅ Core feature | ❌ No           | ❌ No         |
| **Model distillation**     | ✅ SetFit/M2V   | ❌ No         | ✅ Yes          | ❌ No           | ⚠️ Manual    |
| **Learning curve**         | Low            | Medium       | High           | Low            | Low          |
| **Best for**               | Prototyping    | Production   | Research       | Structured gen | General ML   |

**When to choose sieves:**

- You want to implement a document processing/AI use case
- You want to prototype quickly without committing to a specific backend
- You need end-to-end document AI workflow (parsing → processing → distillation)
- You value a unified interface over framework-specific features

**When to choose alternatives:**

- You're looking for a fully featured LLM framework
- You want to implement a chat bot or RAG use case
- **LangChain**: Already deeply integrated in your production stack
- **DSPy**: Research projects requiring custom optimization algorithms
- **Outlines**: Simple structured generation without pipeline needs
- **Transformers**: Maximum flexibility and fine-grained control

## Supported Models

`sieves` works with multiple NLP frameworks. Here's how to create models for each:

#### DSPy

See [docs](https://dspy.ai/).

```python
import dspy
import os

model = dspy.LM(
    "anthropic/claude-4-5-haiku",
    api_key=os.environ["ANTHROPIC_API_KEY"]
)
```
---

#### Outlines

See [docs](https://dottxt-ai.github.io/outlines/welcome/).

```python
import outlines
import transformers

model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
model = outlines.models.from_transformers(
    transformers.AutoModelForCausalLM.from_pretrained(model_name),
    transformers.AutoTokenizer.from_pretrained(model_name)
)
```

---

#### Transformers (Zero-Shot Classification Pipelines)

See [docs](https://huggingface.co/tasks/zero-shot-classification).

```python
import transformers

model = transformers.pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33",
    device=0
)
```

---

#### LangChain

See [docs](https://docs.langchain.com/). E.g. with an OpenAI model:
```python
from langchain_openai import ChatOpenAI
import os

model = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-5-mini",
    temperature=0
)
```
---

#### GLiNER2

See [docs](https://fastino.ai/docs/gliner-2-overview).


**Basic usage:**
```python
import gliner2

model = gliner2.GLiNER2.from_pretrained("fastino/gliner2-base-v1")
```

See the [Model Setup Guide](https://sieves.ai/guides/models) for more details and troubleshooting.

## Core Concepts

sieves is built on three key abstractions:

- **`Doc`**: Represents a document with text, metadata, and processing results
- **`Task`**: A processing step (classification, extraction, summarization, etc.)
- **`Pipeline`**: Orchestrates tasks with caching, serialization, and observability

[→ Read the architecture guide](https://sieves.ai/guides/custom_tasks#understanding-the-architecture) for details on bridges, engines, and internals.

## Get Started

<div align="center">

📖 **[Read the guides →](https://sieves.ai/guides/getting_started)**
Start with the 5-minute tutorial

🎯 **[Browse examples →](https://sieves.ai/examples)**
Real-world use cases and patterns

🤝 **[Join discussions →](https://github.com/mantisai/sieves/discussions)**
Ask questions, share projects

⭐ **[Star on GitHub](https://github.com/mantisai/sieves)**
Stay updated with new features

</div>

## Frequently Asked Questions

<details>
  <summary><b>Why "sieves"?</b></summary>

  Filtering an LLM's potentially endless stream of unstructured text into a structured format is like sieving water to capture gold nuggets. That's where the name comes from.

</details>

<details>
  <summary><b>Why not just prompt an LLM directly?</b></summary>

  - Validated outputs: Structured results with type checking
  - Observable pipelines: Debug each stage
  - Backend flexibility: Switch models without rewriting
  - Built-in tooling: Caching, serialization, optimization

</details>

<details>
  <summary><b>How do I set up models?</b></summary>

  See the [Model Setup Guide](https://sieves.ai/guides/models) for framework-specific examples.

  Quick example: `model = dspy.LM("claude-3-haiku-20240307", api_key=os.environ["ANTHROPIC_API_KEY"])`

</details>

<details>
  <summary><b>Can I use local models?</b></summary>

  Yes! Via Ollama, vLLM, or Transformers directly. [See guide](https://sieves.ai/guides/models#local-models).

</details>

<details>
  <summary><b>Is sieves production-ready?</b></summary>

  Beta status: API stable within minor versions, well-tested, used in real projects.
  Pin your version: `pip install "sieves==0.x.*"`

</details>

## Attribution

`sieves` is inspired by [spaCy](https://spacy.io/) and [spacy-llm](https://github.com/explosion/spacy-llm).

> <a href="https://www.flaticon.com/free-icons/sieve" title="sieve icons">Sieve icons created by Freepik - Flaticon</a>.
