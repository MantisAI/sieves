# Model Setup

This guide explains how to set up models for use with `sieves` across different frameworks and providers.

## Overview

`sieves` supports multiple a bunch of language model frameworks - each allowing different usage modes, pros and cons, and
supporting different use cases.

This table attempts to capture essential properties of each supported framework, including a very coarse categorization
of the frameworks flexibility and efficiency.

| Framework                                   | Flexibility | Efficiency  | Supported Model (Types)                   | Remote Model Support |
|---------------------------------------------|-------------|-------------|-------------------------------------------|----------------------|
| **DSPy**                                    | 🟢🟢🟢      | 🟢          | Via LiteLLM: most remote and local models | ✅ Yes                 |
| **Outlines**                                | 🟢🟢🟢      | 🟢          | All common local models                   | ✅ Yes                 |
| **Transformers (zero-shot classification)** | 🟢          | 🟢🟢🟢      | All Hugging Face models                   | ❌ No                  |
| **LangChain**                               | 🟢🟢🟢      | 🟢          | Most remote and local models              | ✅ Yes                 |
| **GLiNER**                                  | 🟢🟢        | 🟢🟢        | Specialized GliNER models                 | ❌ No                  |

One useful perspective on these frameworks is to view them on a spectrum `flexibility <-> efficiency`. Large LLMs, and
the model frameworks who use them, are extremely flexible, but slower. Smaller, specialized models, and their
corresponding frameworks, are less flexible, but more efficient in what they're doing.

## Framework-Specific Setup

### DSPy

> DSPy is a declarative framework for building modular AI software. It allows you to iterate fast on structured code, rather than brittle strings, and offers algorithms that compile AI programs into effective prompts and weights for your language models, whether you're building simple classifiers, sophisticated RAG pipelines, or Agent loops.

See [docs](https://dspy.ai/).

#### Cloud Providers

**Anthropic Claude:**
```python
import dspy
import os

model = dspy.LM(
    "anthropic/claude-4-5-haiku",
    api_key=os.environ["ANTHROPIC_API_KEY"]
)
```

**OpenAI:**
```python
import dspy
import os

model = dspy.LM(
    "gpt-5-mini",
    api_key=os.environ["OPENAI_API_KEY"]
)
```

**OpenRouter (supports many models):**
```python
import dspy
import os

model = dspy.LM(
    "openrouter/google/gemini-2.5-flash-lite-preview-09-2025",
    api_base="https://openrouter.ai/api/v1/",
    api_key=os.environ["OPENROUTER_API_KEY"]
)
```

#### Local Models via Ollama

**Requirements:**
1. Install [Ollama](https://ollama.com/)
2. Pull a model: `ollama pull llama3`

**Usage:**
```python
import dspy

model = dspy.LM(
    "ollama/qwen3",
    api_base="http://localhost:11434"
)
```

#### Local Models via vLLM

**Requirements:**
1. Install vLLM: `pip install vllm`
2. Start vLLM server:
   ```bash
   vllm serve HuggingFaceTB/SmolLM-135M-Instruct --port 8000
   ```

**Usage:**
```python
import dspy

model = dspy.LM(
    "openai/HuggingFaceTB/SmolLM-135M-Instruct",
    api_base="http://localhost:8000/v1"
)
```

---

### Outlines

> Outlines is a Python library that allows you to use Large Language Model in a simple and robust way (with structured generation). It is built by .txt, and is already used in production by many companies.

See [docs](https://dottxt-ai.github.io/outlines/welcome/).


**Basic usage:**
```python
import outlines
import transformers

model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
model = outlines.models.from_transformers(
    transformers.AutoModelForCausalLM.from_pretrained(model_name),
    transformers.AutoTokenizer.from_pretrained(model_name)
)
```

**With GPU:**
```python
import outlines
import transformers

model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = outlines.models.from_transformers(
    transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically use GPU
        torch_dtype="auto"   # Use optimal dtype
    ),
    transformers.AutoTokenizer.from_pretrained(model_name)
)
```

---

### Transformers (Zero-Shot Classification)

> Transformers acts as the model-definition framework for state-of-the-art machine learning models in text, computer vision, audio, video, and multimodal model, for both inference and training.
> It centralizes the model definition so that this definition is agreed upon across the ecosystem.

See [docs](https://huggingface.co/tasks/zero-shot-classification).

**Basic usage:**
```python
import transformers

model = transformers.pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33"
)
```

**With GPU:**
```python
import transformers

model = transformers.pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33",
    device=0  # Use first GPU
)
```

---

### LangChain

> LangChain is a framework for building agents and LLM-powered applications. It helps you chain together interoperable components and third-party integrations to simplify AI application development – all while future-proofing decisions as the underlying technology evolves

See [docs](https://docs.langchain.com/).

**OpenAI:**
```python
from langchain_openai import ChatOpenAI
import os

model = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-5-mini",
    temperature=0
)
```

**Anthropic:**
```python
from langchain_anthropic import ChatAnthropic
import os

model = ChatAnthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model="claude-4-5-haiku",
    temperature=0
)
```

**OpenRouter:**
```python
from langchain_openai import ChatOpenAI
import os

model = ChatOpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1/",
    model="google/gemini-3-flash-preview",
    temperature=0
)
```

---

### GLiNER2

> GLiNER 2 is Fastino’s open-source, schema-based information extraction model — a unified architecture for Named Entity Recognition (NER), Text Classification, and Structured Data Extraction in one forward pass.

See [docs](https://fastino.ai/docs/gliner-2-overview).


**Basic usage:**
```python
import gliner2

model = gliner2.GLiNER2.from_pretrained("fastino/gliner2-base-v1")
```

---

## Related Guides

- **[Getting Started](getting_started.md)** - Basic usage of sieves
- **[Task Optimization](optimization.md)** - Improve model performance with prompt optimization
- **[Task Distillation](distillation.md)** - Create faster models from large model outputs

## External Resources

- **[LiteLLM Providers](https://docs.litellm.ai/docs/providers)** - Complete list of supported providers for DSPy
- **[Hugging Face Models](https://huggingface.co/models)** - Browse available models for Outlines/Transformers
- **[Ollama Models](https://ollama.com/library)** - Available models for local Ollama deployment
- **[LangChain Integrations](https://python.langchain.com/docs/integrations/chat/)** - LangChain chat model providers
