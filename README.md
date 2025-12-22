<img src="https://raw.githubusercontent.com/mantisai/sieves/main/docs/assets/sieve.png" width="230" align="left" style="margin-right:60px" />
<img src="https://raw.githubusercontent.com/mantisai/sieves/main/docs/assets/sieves_sieve_style.png" width="350" align="left" style="margin-right:60px" />

<br clear="all"/>
<br>

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/mantisai/sieves/test.yml)](https://github.com/mantisai/sieves/actions/workflows/test.yml)
![GitHub top language](https://img.shields.io/github/languages/top/mantisai/sieves)
[![PyPI - Version](https://img.shields.io/pypi/v/sieves)]((https://pypi.org/project/sieves/))
![PyPI - Status](https://img.shields.io/pypi/status/sieves)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)
[![codecov](https://codecov.io/gh/mantisai/sieves/branch/main/graph/badge.svg)](https://codecov.io/gh/mantisai/sieves)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17633730.svg)](https://doi.org/10.5281/zenodo.17633730)

## A Unified Interface for Document AI Applications.

`sieves` provides a stable, framework-agnostic abstraction for building document AI pipelines.
Just as `sqlalchemy` provides a unified interface for interchangeable database drivers, `sieves` offers a consistent API for predictive tasks while allowing you to swap the underlying language model frameworks without changing your core application logic.

This approach recognizes that different LM frameworks excel at different aspects of language model development:
*   [`outlines`](https://github.com/dottxt-ai/outlines) for high-performance, strictly constrained structured generation with local models.
*   [`dspy`](https://github.com/stanfordnlp/dspy) for sophisticated prompt optimization and few-shot example tuning.
*   [`langchain`](https://github.com/langchain-ai/langchain) for broad compatibility with proprietary APIs and existing ecosystems.
*   [`gliner2`](https://github.com/fastino-ai/GLiNER2) or [`transformers`](https://github.com/huggingface/transformers) zero-shot pipelines for specialized, low-latency local inference.

`sieves` unifies the entire workflow:

1.  **Ingestion**: Parsing PDFs, images, and Office docs (via [`docling`](https://github.com/docling-project/docling)).
2.  **Preprocessing**: Intelligent text chunking and windowing (via [`chonkie`](https://github.com/chonkie-inc/chonkie)).
3.  **Prediction**: Zero-shot structured generation using a unified interface.
      Supports multiple backends: [`dspy`](https://github.com/stanfordnlp/dspy), [`langchain`](https://github.com/fastino-ai/GLiNER2), [`outlines`](https://github.com/dottxt-ai/outlines), [`gliner2`](https://github.com/fastino-ai/GLiNER2), [`transformers`](https://github.com/huggingface/transformers) zero-shot classification pipelines
4.  **Distillation**: Distill a specialized local model from zero-shot predictions (via [`setfit`](https://github.com/huggingface/setfit) and [`model2vec`](https://github.com/MinishLab/model2vec)).

Define your task pipeline once, then swap execution engines without rewriting your pipeline logic. Use the task library
to skip having to define tasks from scratch.

> [!WARNING]
> `sieves` is in active development (Beta). The API is stable within minor versions, but we recommend pinning your version for production use.

## Features

- :dart: **Zero Training Required:** Immediate inference using zero-/few-shot models
- :robot: **Unified Generation Interface:** Seamlessly use multiple libraries
  - [`dspy`](https://github.com/stanfordnlp/dspy)
  - [`gliner2`](https://github.com/fastino-ai/GLiNER2)
  - [`langchain`](https://github.com/langchain-ai/langchain)
  - [`outlines`](https://github.com/dottxt-ai/outlines)
  - [`transformer`](https://github.com/huggingface/transformers)
- :arrow_forward: **Observable Pipelines:** Easy debugging and monitoring with conditional task execution
- :hammer_and_wrench: **Integrated Tools:**
  - Document parsing (optional via `ingestion` extra): [`docling`](https://github.com/DS4SD/docling), [`marker`](https://github.com/VikParuchuri/marker)
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
- :floppy_disk: **Persistence:** Save and load pipelines with configurations
- :rocket: **Optimization:** Improve task performance by optimizing prompts and few-shot examples using [DSPy's MIPROv2](https://dspy-docs.vercel.app/api/optimizers/MIPROv2)
- :teacher: **Distillation:** Fine-tune smaller, specialized models using your zero-shot results with frameworks like SetFit and Model2Vec.
  Export results as HuggingFace [`Dataset`](https://github.com/huggingface/datasets) for custom training.
- :recycle: **Caching** to avoid unnecessary model calls

## Quick Start

**1. Install**
```bash
pip install sieves
```
*Requires Python 3.12 (due to dependency constraints in `docling` and `pyarrow`).*

**2. Basic: text classification with a small local model**

```python
import outlines
import transformers
from sieves import Pipeline, tasks, Doc

# Set up model.
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = outlines.models.from_transformers(
    transformers.AutoModelForCausalLM.from_pretrained(model_name),
    transformers.AutoTokenizer.from_pretrained(model_name)
)

# Define task.
task = tasks.Classification(labels=["science", "politics"], model=model)

# Define pipeline with the classification task.
pipeline = Pipeline(task)

# Define documents to analyze.
doc = Doc(text="The new telescope captures images of distant galaxies.")

# Run pipeline and print results.
results = list(pipeline([doc]))
docs = list(pipeline([doc]))
# The `results` field contains the structured task output.
print(docs[0].results) # {'Classification': [('science', 1.0), ('politics', 0.0)]}
# The `meta` field contains more information helpful for observability and debugging, such as raw model output and token count information.
print(docs[0].meta)    # {'Classification': {
                       #    'raw': ['{ "science": 1.0, "politics": 0 }'],
                       #    'usage': {'input_tokens': 2, 'output_tokens': 2, 'chunks': [{'input_tokens': 2, 'output_tokens': 2}]}}, 'usage': {'input_tokens': 2, 'output_tokens': 2}
                       #  }
```

**3. Advanced: End-to-end document AI with a hosted LLM**

This example demonstrates the full power of `sieves`: parsing a PDF, chunking it, and extracting structured data (equations) using a remote LLM via DSPy.

*Requires `pip install "sieves[ingestion]"`*

```python
import dspy
import os
import pydantic
import chonkie
import tokenizers
from sieves import tasks, Doc

# Define which schema of entity to extract.
class Equation(pydantic.BaseModel, frozen=True):
    id: str = pydantic.Field(description="ID/index of equation in paper.")
    equation: str = pydantic.Field(description="Equation as shown in paper.")

# Setup DSPy model.
model = dspy.LM(
    "openrouter/google/gemini-3-flash-preview",
    api_base="https://openrouter.ai/api/v1/",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

# Build pipeline: ingest -> chunk -> extract.
pipeline = (
    tasks.Ingestion() +
    tasks.Chunking(chonkie.TokenChunker(tokenizers.Tokenizer.from_pretrained("gpt2"))) +
    tasks.InformationExtraction(entity_type=Equation, model=model)
)

# Define docs to analyze.
doc = Doc(uri="https://arxiv.org/pdf/1204.0162")

# Run pipeline.
results = list(pipeline([doc]))

# Print esults.
for equation in results[0].results["InformationExtraction"]:
    print(equation)
```
This gives us:
```
id='(1)' equation="the observer measures not the linear but angular ... both cars are near the stop sign."
id='(3)' equation='\\omega(t) = \\frac{r_0 v(t)}{r_0^2 + x(t)^2}'
id='(4)' equation='\\tan \\alpha(t) = \\frac{x(t)}{r_0}'
id='(5)' equation='x(t) = \\frac{a_0 t^2}{2}'
id='(6)' equation="\\frac{d}{dt} f(t) = f'(t)"
id='(7)' equation='\\omega(t) = \\frac{a_0 t}{r_0} \\left( 1 + \\frac{a_0^2 t^4}{4 r_0^2} \\right)^{-1}'
id='(8)' equation='x(t) = x_0 + v_0 t + \\frac{1}{2} a t^2'
```

**[Read the guides]**(https://sieves.ai/guides/getting_started)

---

## Why `sieves`?

Building Document AI prototypes usually involves gluing together disparate tools: one library for PDF parsing, another
for chunking, a third for LLM interaction, another one for distillation, and so on.
Switching from one model/framework stack, e.g., using `Outlines` with a local model, to a different one, e.g.
`LangChain` with a closed vendor LLM, often requires rewriting core logic and boilerplate.

`sieves` solves this by providing a **vertical stack** optimized for Document AI.

**Best for:**
*   ✅ **Document AI**: End-to-end pipelines from raw file to structured data.
*   ✅ **Rapid Prototyping**: Validate ideas quickly with zero-shot models; no training data needed.
*   ✅ **Backend Flexibility**: Switch between Local (GLiNER, Outlines) and Remote (DSPy, LangChain) execution instantly.
*   ✅ **Observability**: Built-in inspection of intermediate steps (chunks, prompts).

**Not for:**
*   ❌ Chatbots or conversational agents.
*   ❌ Simple, one-off LLM completion calls.

### Feature Comparison

| Feature                 | `sieves`                 | `langchain`        | `dspy`                     | `outlines`            | `transformers` | `gliner2`    |
|:------------------------|:-------------------------|:-------------------|:---------------------------|:----------------------|:---------------|:-------------|
| **Primary Focus**       | **Document AI**          | General LLM apps   | Declarative LM development | Structured generation | Modeling       | Extraction   |
| **Backend Support**     | **Universal**            | Own ecosystem      | Own ecosystem              | Own ecosystem         | Own ecosystem  | Specialized  |
| **Document Parsing**    | **Built-in**             | Tool integrations  | ❌ No                       | ❌ No                  | ❌ No           | ❌ No         |
| **Structured Output**   | **Unified Pydantic API** | Framework-specific | Framework-specific         | Core feature          | ⚠️ Limited     | Core feature |
| **Prompt Optimization** | **DSPy Integration**     | ❌ No               | ✅ Core feature             | ❌ No                  | ❌ No           | ❌ No         |
| **Model Distillation**  | **`setfit`/`model2vec`** | ❌ No               | ✅ Yes                      | ❌ No                  | ⚠️ Manual      | ❌ No         |

## Core Concepts

*   **`Doc`**: The atomic unit of data. Holds raw text, metadata, parsed content, and extraction results.
*   **`Task`**: A functional step in the pipeline (e.g., `Ingestion`, `Chunking`, `NER`, `Classification`).
*   **`Pipeline`**: A composable sequence of tasks that manages execution flow, caching, and state.

## Supported Backends

`sieves` allows you to bring your own model backend. We support:

*   **DSPy**: For optimizing prompts and working with remote/local models via LiteLLM.
*   **Outlines**: For strictly constrained structured generation with local models.
*   **LangChain**: For broad compatibility with the LangChain ecosystem.
*   **GLiNER2**: For high-performance, small-model Named Entity Recognition.
*   **Transformers**: For standard Hugging Face zero-shot classification pipelines.

See the [Model Setup Guide](https://sieves.ai/guides/models) for configuration details.

## Installation

```bash
pip install sieves
```

**Optional extras:**
```bash
pip install "sieves[ingestion]"  # PDF/DOCX parsing (docling, marker)
pip install "sieves[distill]"    # Model distillation (setfit, model2vec)
```

## Community & Support

<div align="center">

📖 **[Documentation](https://sieves.ai/)** •
❓ **[Chat with the `sieves` DeepWiki](https://deepwiki.com/MantisAI/sieves)** •
🤝 **[Discussions](https://github.com/mantisai/sieves/discussions)**

</div>

## Attribution

`sieves` is inspired by the design philosophy of [spaCy](https://spacy.io/) and [spacy-llm](https://github.com/explosion/spacy-llm).

> <a href="https://www.flaticon.com/free-icons/sieve" title="sieve icons">Sieve icons created by Freepik - Flaticon</a>.
