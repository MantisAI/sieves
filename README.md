<img src="docs/assets/sieve.png" width="200" height="200" align="left" style="margin-right:60px" />

<h1 align="left">
<br>
<span style="font: bold 38pt'Courier New';">sieves</span>
<br>
<br>
<br>
</h1>



[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/mantisai/sieves/test)](https://github.com/mantisai/sieves/actions/workflows/test.yml)
![GitHub top language](https://img.shields.io/github/languages/top/mantisai/sieves)
[![pypi Version](https://img.shields.io/pypi/v/sieves.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/sieves/)
![PyPI - Status](https://img.shields.io/pypi/status/sieves)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sieves)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)
[![codecov](https://codecov.io/gh/mantisai/sieves/branch/main/graph/badge.svg)](https://codecov.io/gh/mantisai/sieves)

## Sieve your data: zero-shot NLP made easy.

`sieves` enables zero- and few-shot NLP tasks with structured generation. With no training required, you can quickly 
prototype NLP tasks while ensuring reliable, unified output formats.

`sieves` allows running NLP tasks off the bat with zero- and few-shot models, leveraging structured generation methods. 
- No training needed
- Consistent structured output
- Immediate productivity using built-in tasks
- Flexible customization for your own tasks

A simple example, using `outlines` for classification:
```python
import outlines

from sieves import Pipeline, engines, tasks, Doc

# 1. Define documents by text or URI.
docs = [Doc(text="Special relativity applies to all physical phenomena in the absence of gravity.")]

# 2. Create engine responsible for generating structured output.
model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
engine = engines.outlines_.Outlines(model=outlines.models.transformers(model_name))

# 3. Create pipeline with tasks.
pipe = Pipeline(
    [
        # 4. Run classification on provided document.
        tasks.predictive.Classification(labels=["science", "politics"], engine=engine),
    ]
)

# 5. Run pipe and output results.
docs = list(pipe(docs))
print(docs[0].results["Classification"])
```

<details>
  <summary><b>A more involved example</b></summary>

Here we parse a PDF with `docling`, chunk it with `chonkie`, and classify it with `gliclass`:
```python
import transformers
import gliclass
import chonkie
import tokenizers

from sieves import Pipeline, engines, tasks, Doc

# 1. Define documents by text or URI.
docs = [Doc(uri="https://arxiv.org/pdf/2408.09869")]

# 2. Create engine responsible for generating structured output.
model_name = "knowledgator/gliclass-small-v1.0"
pipeline = gliclass.ZeroShotClassificationPipeline(
    gliclass.GLiClassModel.from_pretrained(model_name),
    transformers.AutoTokenizer.from_pretrained(model_name),
    classification_type="multi-label",
)
engine = engines.glix_.GliX(model=pipeline)
    
# 3. Create pipeline with tasks.
pipe = Pipeline(
    [
        # 4. Add document parsing task.
        tasks.parsers.Docling(),
        # 5. Add chunking task to ensure we don't exceed our model's context window.
        tasks.chunkers.Chonkie(chonkie.TokenChunker(tokenizers.Tokenizer.from_pretrained(model_name))),
        # 6. Run classification on provided document.
        tasks.predictive.Classification(task_id="classifier", labels=["science", "politics"], engine=engine),
    ]
)

# 7. Run pipe and output results.
docs = list(pipe(docs))
print(docs[0].results["classifier"])
```
</details>

### Features

- :dart: **No training:** All tasks run with structured zero-/few-shot models 
- :robot: **Integration and unified usage of multiple structured generation libraries**:
  - [`outlines`](https://github.com/dottxt-ai/outlines)
  - [`dspy`](https://github.com/stanfordnlp/dspy)
  - [`gliner`](https://github.com/Knowledgator/GLiClass) and [`gliclass`](https://github.com/Knowledgator/GLiClass)
  - [`transformer`] zero-shot pipelines
  - TBD: [`ollama`](https://github.com/ollama/ollama)
- :arrow_forward: **Pipeline-based system** for easy observability and debugging
- :hammer_and_wrench: **Integrated utilities** convenient in an NLP pipeline
  - File parsing: [`docling`]()
  - Chunking: [`chonkie`]()
  - TBD: export to [`datasets`]() dataset for easy fine-tuning 
- :label: Prebuilt tasks ready to use out-of-the-box
  - Classification
  - TBD: NER, entity linking, information extraction, summarization, translation, ...
- :floppy_disk: Serializability (TBD)


### Why `sieves`?

Even with generative AI, structured outputs and observability remain essential. 

In many real-life use-cases, there is a need for prototyping solutions that work with very little data. Generative 
language models fill this gap nicely. Getting them to output clean, structured data can be tricky though. This is addressed by tooling for 
structured/steered language model output: [`outlines`](https://github.com/dottxt-ai/outlines), [`dspy`](https://github.com/stanfordnlp/dspy), [`ollama`](https://github.com/ollama/ollama) and others.
Each of these libraries has its own patterns, pros and cons. `sieves` offers a unified interface for usage, in- and output. 

Beyond that, we often encounter the same set of steps to get a prototype off the ground - such as parsing and chunking 
documents, exporting results in a dataset format that can be used to fine-tune models, and experimenting with different ways of 
prompting/encouraging the model to structure its output. All of these things are solved by existing libraries in the NLP
ecosystem (think [`docling`](https://github.com/DS4SD/docling) or Hugging Face's 
[`datasets`](https://github.com/huggingface/datasets). 

**What `sieves` aims to do** is to bundle all of this into one library that enables you to quickly and painlessly build 
prototypes of NLP-heavy applications. Built on zero- and few-shot models and a number of pre-implemented tasks you can 
use right without having to build your own.  

`sieves` has been partially inspired by [`spacy`](), especially [`spacy-llm`](https://github.com/explosion/spacy-llm). 

### Core Abstractions

We intend to keep the API is minimalist and modular as possible. When using `sieves`, you'll need to know about the 
following abstrations:

#### Pipeline
Responsible for coordinating the execution of tasks. Typical interactions with a `Pipeline` instance:
- Instatiate pipeline
- Add tasks to pipeline
- Run pipeline
- Serialize pipeline (config)

#### Doc
A `Doc` instance represents one document and contains information about its content, URI (if available) and potential 
meta-information. The `sieves` workflow pipeline revolves around `Doc` instances - information is propagated throughout
the pipeline via `Doc` instances being passed from task to task.

> [!IMPORTANT]  
> The length of your `Doc`s might exceed your models context length. Make sure you run a chunking task in your pipeline
> to avoid running into problems with the context length.

#### Engine
An `Engine` wraps an existing library designed for structured generation with zero-/few-shot models. It coordinates 
setting up executing prompts and fetching output from the model and passes it back to be converted into a unified data 
format.

> [!WARNING]  
> Engines might fail in producing structured output. Validation ensures you won't end up with improperly structured 
> results, but you might end up with a pipeline failure. The risk for this correlates positively with the complexity of 
> the expected response type and negatively with the capability of the used model.

#### Task
A `Task` implements a given NLP task (such as classification, NER, information extraction, ...) for at least one engine.
In itself a `Task` implementation is usually very concise, as most of the logic will be implemented in `Bridge` classes,
which implement task logic with respect to a certain engine. I.e.: one `Task` provides one or more `Bridge` classes,
which implement a task for one `Engine`.

> [!TIP]
> Some tasks might work better with one engine than the other. When implementing your own tasks, make sure to experiment
> to figure out which engine delivers the best performance.

#### Bridge
A `Bridge` class connects one `Task` with one `Engine`. E.g.: a classification task will require a different 
implementation when run against `outlines` then when run against `dspy` or `ollama`. This is the responsibility of a 
`Bridge` implementation. `Bridge` implements functionality such as specifying a prompt template or the prompt's expected
output type.

Note that the `Bridge` abstraction is not relevant if you're just using `sieves`, as it's a part of a `Task`s 
composition. It is however relevant when implementing your own task, as you will need at least one `Bridge` for any 
given `Task`. 

## Frequently Asked Questions

<details>
  <summary><b>Show FAQs</b></summary>

### What's the meaning behind the name?

Originally, `sieves` was intended for information extraction. The name comes from [gold panning](https://en.wikipedia.org/wiki/Gold_panning): 
run your raw data through a sieve to obtain structured, refined “gold.”

### Why not just prompt an LLM directly?

You can - but `sieves` offers:
    - Structured data output. Zero-/few-shot LLMs can be finicky without guardrails or parsing.
    - A step-by-step pipeline, making it easier to debug and track each stage.
 

### Why use `sieves` and not a structured generation library like `outlines`?

This is not either-or - `sieves` includes `outlines` (among others), plus:
    - A uniform input/output format
    - Prebuilt NLP tasks
    - Pipeline logic with chunking and chunk consolidation
    - Easy switching between structured generation engines
    - Tools for file parsing, chunking, exporting data for fine-tuning

</details>

---

> Source for `sieves` icon:
> <a href="https://www.flaticon.com/free-icons/sieve" title="sieve icons">Sieve icons created by Freepik - Flaticon</a>.