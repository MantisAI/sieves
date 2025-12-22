# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Summary

`sieves` is a Python library for rapid, production-minded prototyping of NLP pipelines with zero- and few-shot models and structured outputs. It provides:

- A document-centric pipeline (`Pipeline`) with caching and serialization
- Preprocessing tasks (document ingestion, text chunking)
- Predictive tasks (classification, NER, information extraction, QA, summarization, translation, sentiment analysis, PII masking)
- A unified model interface over structured-generation frameworks (Outlines, DSPy, Instructor, LangChain, Transformers, Ollama, GLiNER, etc.)
- Postprocessing and model distillation helpers

Key packages and concepts: `sieves.data.Doc`, `sieves.pipeline.Pipeline`, `sieves.tasks.*`, `sieves.model_wrappers.*`, `sieves.serialization.Config`.

### Objectives

- Provide reliable, structured outputs with zero/few-shot models
- Make pipelines easy to compose, observe, cache, and serialize
- Support multiple structured-generation model libraries behind one interface
- Enable distillation to smaller, local models for cost/performance optimization

---

## Repository Layout

```
sieves/                    # Core library
├── data/                  # Document model (Doc class)
├── pipeline/              # Pipeline orchestration and caching
├── tasks/                 # Preprocessing, predictive, and postprocessing tasks
│   ├── predictive/        # 8 task types (classification, NER, IE, etc.)
│   └── preprocessing/     # Ingestion and chunking
├── model_wrappers/        # Unified interface to structured generation backends
└── serialization/         # Config and persistence helpers
docs/                      # MkDocs documentation
sieves/tests/              # Comprehensive test suite
pyproject.toml             # Dependencies, metadata, tool config
mkdocs.yml                 # Documentation build config
AGENTS.md                  # This file (Claude Code guidelines)
```

---

## Installation & Setup

**Python requirement:** 3.12 (exact version required)

### Using `uv` (preferred)

```bash
uv sync                              # Base installation
uv sync --extra distill              # Add distillation (SetFit, Model2Vec)
uv sync --extra ingestion            # Add document parsing (Docling, Marker, NLTK)
uv sync --all-extras                 # Everything (includes test tools)
```

**Note:** As of recent updates, all supported model libraries (Outlines, DSPy, LangChain, Transformers, GLiNER2) are now core dependencies included in the base installation.

### Using pip (editable)

```bash
pip install -e .                     # Base
pip install -e ".[distill,ingestion]"  # With extras
```

### Environment Variables

Set only what you need:

```bash
OPENAI_API_KEY                       # For OpenAI models
ANTHROPIC_API_KEY                    # For Claude models
OPENROUTER_API_KEY                   # For OpenRouter
OLLAMA_HOST                          # For local Ollama models (e.g., http://localhost:11434)
```

---

## Development Commands

All commands assume `uv` is available; adjust for `pip` + venv if needed.

### Verification & Quality

```bash
# Install development dependencies (includes test tools)
uv sync --all-extras

# Run type checker (strict mode)
uv run mypy sieves

# Run linter
uv run ruff check .

# Auto-fix safe lint issues
uv run ruff check . --fix

# Format code
uv run black .

# Run full quality pipeline
uv run mypy sieves && uv run ruff check . && uv run black --check .
```

### Testing

```bash
# Run all tests
uv run pytest -q

# Run with coverage report
uv run pytest --cov=sieves

# Run a specific test file
uv run pytest sieves/tests/test_doc.py -v

# Run tests matching a pattern (e.g., classification tasks)
uv run pytest -k classification -v

# Run fast tests only (skip slow tests)
uv run pytest -m "not slow"
```

### Documentation

```bash
# Build and serve docs locally
uv run mkdocs serve

# Build static docs
uv run mkdocs build --strict
```

### Import & Sanity Check

```bash
# Verify installation
uv run python -c "import sieves; print(sieves.__name__)"
```

---

## Architecture Overview

### Core Abstractions

1. **Doc** (`sieves.data.Doc`)
   - Container for text, URI, chunks, images, and processing results
   - Auto-chunks text on initialization using Chonkie
   - `results` dict stores task outputs keyed by task ID
   - Supports image inputs via PIL (stored in `images` field)

2. **Pipeline** (`sieves.pipeline.Pipeline`)
   - Orchestrates sequential task execution
   - Caches by document hash (text or URI)
   - Supports composition via `+` operator
   - Serializable via `dump()`/`load()`

3. **Task** (`sieves.tasks.core.Task`)
   - Base class for all processing steps
   - Subclasses: `PredictiveTask`, `Ingestion`, `Chunking`, `Optimization`
   - Defines `__call__(docs)` for processing
   - Supports conditional execution via `condition` parameter
   - Configurable batching via `batch_size` parameter

4. **ModelWrapper** (`sieves.model_wrappers.core.ModelWrapper`)
   - Generic interface to structured generation frameworks
   - Implementations: DSPy (v3), Outlines (default), LangChain, Transformers, GLiNER2
   - Each model wrapper implements `build_executable()` to compile prompts
   - All supported model libraries are now core dependencies (no longer optional)

5. **Bridge** (`sieves.tasks.predictive.bridges.Bridge`)
   - Connects tasks to model wrappers
   - Defines prompt templates (Jinja2-based)
   - Handles output schema and parsing
   - Specialized bridges like `GliNERBridge` can be shared across tasks

6. **ModelSettings** (`sieves.model_wrappers.types.ModelSettings`)
   - Configures structured generation behavior
   - Fields: `init_kwargs`, `inference_kwargs`, `inference_mode`, `strict`, batch settings
   - `strict=True`: raises on inference failure; `False`: yields None for failed docs

### Data Flow

```
Doc creation
  ↓
Pipeline([task1, task2, ...])
  ↓
pipe(docs)  # Execute all tasks
  ↓
For each task:
  - Check cache (by text/URI hash)
  - Get/create model wrapper executable
  - Run inference via Bridge
  - Parse and store results in Doc.results[task_id]
  ↓
Return docs with populated results
```

### Supported Model wrappers

| ModelWrapper | Type | Inference Modes | Notes |
|---|---|---|---|
| **Outlines** | Structured generation | text, choice, regex, json | Default; JSON schema constrained |
| **DSPy** (v3) | Modular prompting | predict, chain_of_thought, react, module | Few-shot, optimizer support (MIPROv2) |
| **LangChain** | LLM wrapper | structured | Chat models, tool calling |
| **Transformers** | Direct inference | zero_shot_classification | HuggingFace zero-shot classification pipeline |
| **GLiNER2** | Specialized NER | (specialized) | Domain-specific NER, zero-shot entity recognition |

---

## Coding Standards

Enforced via CI pipeline:

- **Type checking:** MyPy strict mode (`[tool.mypy]` in pyproject.toml)
  - Disallow untyped defs, implicit optional, untyped decorators
  - All imports must be resolvable or ignored

- **Linting:** Ruff with E, F, I, UP, D rules
  - Docstrings follow PEP 257 (single-line preferred; no blank line after summary if multi-line)
  - Max McCabe complexity: 10

- **Formatting:** Black (line length 120)
  - Auto-applied; check with `black --check .`

- **Best practices:**
  - No one-letter variables (except `i` in loops)
  - Minimal, focused changes
  - Gate optional-dependency imports behind `try/except` or feature checks

---

## Extension Points

### Adding a New Predictive Task

1. Create module: `sieves/tasks/predictive/<task_name>/`
2. Implement `core.py`:
   - Subclass `PredictiveTask`
   - Define `__call__` for execution
3. Create `bridges.py`:
   - Subclass `Bridge` for each supported model wrapper
   - Use generic bridges (e.g. `GliNERBridge`) if applicable
   - Define prompt template (Jinja2), output schema (Pydantic), extraction/parsing logic
4. Export in `sieves/tasks/predictive/__init__.py`
5. Add tests under `sieves/tests/tasks/predictive/`
6. Add docs to `docs/tasks/`:
   - Include usage examples with snippets from `sieves/tests/docs/`
   - Link to third-party libraries

### Adding a New ModelWrapper

1. Create file: `sieves/model_wrappers/<model_wrapper_name>_.py`
2. Subclass appropriate base (e.g., `PydanticModelWrapper` for schema-aware generation)
3. Implement `build_executable(signature, **kwargs)` → callable
4. Advertise `inference_modes` property
5. Add to `ModelType` enum in `model_type.py`
6. Ensure `serialize()/deserialize()` work with `Config`
7. Add tests and docs (with snippets)

### Custom Preprocessing

- Create under `sieves/tasks/preprocessing/<type_>/`
- Subclass `Task` or specialized base (e.g., `Chunking`)
- Examples: custom chunkers, PDF parsers, text normalizers
- **Built-in chunking**: Uses Chonkie framework (token-based) or NaiveChunker (interval-based)
- **Built-in ingestion**: Docling (default) and Marker converters for PDF/DOCX parsing

### Few-Shot Examples

- Define as Pydantic models matching prompt signature
- Pass via `fewshot_examples` parameter to task
- Model wrappers handle serialization/batching compatibility

### Model Optimization

- Use `sieves.tasks.optimization.Optimization` with DSPy's MIPROv2
- Requires labeled examples for few-shot tuning

### Model Distillation

- Call `task.to_hf_dataset(docs, threshold=...)` to export results to HuggingFace dataset format
- Use `task.distill(dataset, framework="setfit", ...)` to train smaller model
- Supported frameworks: SetFit, Model2Vec
- Available for classification, NER, and other predictive tasks

---

## Caching & Performance

- **Document-level caching:** Pipeline hashes documents by `hash(doc.text or doc.uri)`; cache stores results
- **Disable when needed:** `Pipeline(use_cache=False)`
- **Batch processing:** Configure `batch_size` in task initialization or ModelSettings (−1 = batch all)
- **Streaming:** Tasks accept `Iterable[Doc]` for lazy evaluation on large corpora
- **Conditional execution:** Use `condition` parameter on tasks to filter documents: `task(docs, condition=lambda d: len(d.text) > 100)`
- **Observability:** Loguru logging during execution; access cache stats via pipeline
- **Progress bars:** Configurable via task parameters (can be disabled)

---

## Observability & Serialization

- **Logging:** `loguru` integrated; logs task execution and model wrapper calls.
- **Raw Model Outputs:** Captured in `doc.meta[task_id]['raw']` as a list of raw responses per chunk when `include_meta=True` (default).
- **Token Usage Tracking:**
  - Tracked across the entire pipeline and aggregated in `doc.meta['usage']`.
  - Also available per task in `doc.meta[task_id]['usage']`.
  - Includes `input_tokens` and `output_tokens`.
  - Uses native metadata for DSPy/LangChain and approximate estimation for other backends.
- **Pipeline persistence:**
  ```python
  pipe.dump("pipeline.yml")                        # Save config.
  loaded = Pipeline.load("pipeline.yml", task_kwargs)  # Reload with model kwargs.
  ```
- **Document persistence:** Use pickle (models not serialized).
- **Config format:** YAML-compatible via `sieves.serialization.Config`.

---

## Guardrails for Agents

### Do

- Adhere to typing and lint rules; run mypy/ruff/black before proposing changes
- Keep patches minimal and focused; avoid unrelated refactors
- Respect optional dependencies; gate ingestion/distillation imports behind extras (model libraries are now core)
- Update docs (`docs/`) if you add public features
  - Include introduction and usage examples
  - Use snippets from `sieves/tests/docs/` to ensure code is tested
- Write tests for new functionality
- Consider conditional execution and error handling (`strict`) for robust pipelines

### Don't

- Commit secrets, API keys, or credentials
- Perform destructive operations (mass renames, file deletions) without explicit instruction
- Change public APIs casually; ensure backward compatibility or document breaking changes
- Modify CI/GitHub Actions without guidance

### Escalate

- Network calls to external services or model downloads
- Installing system packages (Tesseract, CUDA, etc.)
- Large dependency changes (adding/removing major model libraries)
- Breaking changes to public APIs

---

## Pre-PR Verification Checklist

Before proposing changes, ensure:

- [ ] Build succeeds: `uv sync [--extra test]` and `uv run python -c "import sieves"`
- [ ] Type checking passes: `uv run mypy sieves`
- [ ] Linting passes: `uv run ruff check .`
- [ ] Formatting correct: `uv run black .` (check with `black --check .`)
- [ ] Tests pass: `uv run pytest` (run full suite or targeted tests)
- [ ] New code covered by tests where practical
- [ ] Docs updated for new public APIs (if applicable)

---

## Known Constraints & Limitations

- Some models do not support batching or few-shotting uniformly; bridge logic handles compatibility
- Optional extras gate heavy dependencies (Docling, Marker for ingestion; SetFit, Model2Vec for distillation)
- **All model libraries** (Outlines, DSPy, LangChain, Transformers, GLiNER2) are now **core dependencies**
- Serialization excludes complex third-party objects (models, converters); must pass at load time
- Ingestion tasks may require system packages (Tesseract for OCR, etc.)
- Python 3.12 exact version required (not 3.12+)

---

## Useful References

- **Code documentation:** `docs/` (MkDocs) and module docstrings
- **ModelWrapper guides:** `docs/model_wrappers/`
- **Task guides:** `docs/tasks/`
- **Getting started:** `docs/guides/getting_started.md`
- **README:** Project overview and quick-start examples
- **API exports:** `sieves/__init__.py`

---

## Quick Start for Agents

### Debugging a Test Failure

```bash
uv run pytest sieves/tests/test_name.py -vv       # Verbose output
uv run pytest sieves/tests/test_name.py --pdb     # Drop into debugger on failure
```

### Understanding Code Flow

1. Start at public API: `sieves/__init__.py`
2. Follow the core classes: `data.Doc` → `pipeline.Pipeline` → `tasks.Task` → `model_wrappers.ModelWrapper`
3. Look at examples: `examples/` directory
4. Check task-specific bridges: `sieves/tasks/predictive/<task>/bridges.py`

### Adding a Quick Test

Create `sieves/tests/test_my_feature.py`:

```python
import pytest
from sieves import Doc, Pipeline, tasks

def test_my_feature():
    doc = Doc(text="Test text")
    # Write test logic
    assert True
```

Then run: `uv run pytest sieves/tests/test_my_feature.py -v`

---

## Version & CI

- **Versioning:** Dynamic via `setuptools-scm`; automated from git tags
- **CI:** GitHub Actions (`.github/workflows/`) runs tests on all PRs
- **Status badges:** See README for latest build/coverage status

---

## Recent Major Updates

Key changes that affect development (last ~2-3 months):

1. **Token Counting and Raw Output Observability** - Implemented comprehensive token usage tracking (input/output) and raw model response capturing in `doc.meta`. Usage is aggregated per-task and per-document.
2. **Information Extraction Single/Multi Mode** - Added `mode` parameter to `InformationExtraction` task for single vs multi entity extraction.
2. **GliNERBridge Refactoring** - Consolidated NER logic into `GliNERBridge`, removing dedicated `GlinerNER` class.
3. **Documentation Enhancements** - Standardized documentation with usage snippets (tested) and library links across all tasks and model wrappers.
4. **All Model wrappers as Core Dependencies** (#210) - Outlines, DSPy, LangChain, Transformers, and GLiNER2 are now included in base installation
5. **DSPy v3 Migration** (#192) - Upgraded to DSPy v3 (breaking API changes from v2)
6. **GliNER2 Migration** (#202) - Migrated from GliNER v1 to GLiNER2 for improved NER performance
7. **ModelSettings Refactoring** (#194) - `inference_mode` moved into ModelSettings (simplified task init)
8. **Conditional Task Execution** (#195) - Added `condition` parameter for filtering docs during execution
9. **Non-strict Execution Support** (#196) - Better error handling; `strict=False` allows graceful failures
10. **Standardized Output Fields** (#206) - Normalized descriptive/ID attribute naming across tasks
11. **Chonkie Integration** - Token-based chunking framework now primary chunking backend
12. **Optional Progress Bars** (#197) - Progress display now configurable per task

---

For questions or updates to these guidelines, refer to maintainers or GitHub issues.
