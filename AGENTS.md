# AGENTS.md

This document is for AI agents and human collaborators working on the sieves codebase. It defines goals, constraints, workflows, tools, and verification steps so agents can operate safely and productively.

If you are an automated agent, follow the Guardrails and Verifications sections strictly. If anything is ambiguous, halt and ask for clarification.

## Project Summary

`sieves` is a Python library for rapid, production‑minded prototyping of NLP pipelines with zero‑ and few‑shot models and structured outputs. It provides:
- A document‑centric pipeline (`Pipeline`) with caching and serialization
- Preprocessing tasks (Ingestion, chunking)
- Predictive tasks (classification, NER, IE, QA, summarization, translation, sentiment, PII masking)
- A unified engine interface over structured‑generation frameworks (Outlines, DSPy, Instructor, LangChain, Transformers, Ollama, GLiNER, etc.)
- Postprocessing and model distillation helpers

Key packages and concepts: `sieves.data.Doc`, `sieves.pipeline.Pipeline`, `sieves.tasks.*`, `sieves.engines.*`, `sieves.serialization.Config`.

## Objectives

- Provide reliable, structured outputs with zero/few‑shot models
- Make pipelines easy to compose, observe, cache, and serialize
- Support multiple structured‑generation engines behind one interface
- Enable distillation to smaller, local models for cost/perf

## Non‑Goals

- End‑to‑end training framework (beyond optional distillation helpers)
- General document storage, orchestration, or MLOps platform

## Repository Layout (high‑level)

- `sieves/` core library
  - `data/` document model
  - `pipeline/` pipeline orchestration and caching
  - `tasks/` preprocessing, predictive, and postprocessing tasks (+ bridges)
  - `engines/` engine wrappers and types
  - `serialization/` config and persistence helpers
- `docs/` user documentation (mkdocs)
- `sieves/tests/` unit tests and test assets
- `pyproject.toml` project metadata and optional extras
- `uv.lock` pinned dependency resolution (for `uv`)

## Environments & Installation

Supported Python: `>=3.10` (tests and typing target 3.11).

Using `uv` (preferred):
- Base: `uv sync`
- With extras:
  - Engines: `uv sync --extra engines`
  - Distill: `uv sync --extra distill`
  - Test tooling/docs: `uv sync --extra test`
  - All extras: `uv sync --all-extras`

Using pip (editable):
- `uv pip install -e .[engines]` (swap `engines` for `distill`, `test` as needed)

Common environment variables (set only what you need):
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, provider‑specific keys supported by LiteLLM/clients
- `OLLAMA_HOST` for local models
- Ingestion/vision may require system packages (e.g., Tesseract) depending on your chosen path

## Quickstart Workflows

Document to results with caching and serialization:
1) Build `Doc` objects: URI, text, or HF datasets via `Doc.from_hf_dataset`
2) Choose an Engine (e.g., Outlines, Instructor, DSPy, GLiNER, Transformers, Ollama)
3) Compose a `Pipeline` with tasks: Ingestion → Chunking → Predictive tasks
4) Run `pipe(docs)`; read results on each `Doc`
5) Persist with `pipe.dump("pipeline.yml")` and pickle/serialize docs

Distillation flow (optional):
- Convert task results to a HF dataset: `task.to_hf_dataset(docs, threshold=...)`
- Call `task.distill(...)` with framework choice and train/val split

## Engines & Tools

The unified `Engine` wraps different backends; some are optional via extras:
- Structured generation: Outlines, Instructor, DSPy, LangChain
- LLM hosting: Transformers, Ollama (vLLM wired but optional/experimental)
- Specialized: GLiNER for NER
- Ingestion & parsing: Docling, Unstructured + Marker PDF
- Chunking: Chonkie

Notes for agents:
- Each predictive task provides a `Bridge` that defines prompt templates, signatures, extraction, consolidation, and integration
- Engines expose `build_executable(...)` that returns a callable to execute structured prompts (often Jinja2‑rendered)
- Few‑shot examples are `pydantic.BaseModel` instances; some engines may not support batching or few‑shotting equally

## Coding Standards

- Type checking: mypy strict (`[tool.mypy]` in `pyproject.toml`)
- Linting: ruff (E, F, I, UP), isort via ruff
- Formatting: black (line length 120)
- Python target version: 3.10 for style/format; 3.11 for mypy config
- Avoid one‑letter variable names; keep changes minimal and focused

## Development Commands

- Install tooling: `uv sync --extra test`
- Lint/format:
  - `uv run ruff check .`
  - `uv run ruff check . --fix` (safe fixes only)
  - `uv run black .`
- Type check: `uv run mypy sieves`
- Tests: `uv run pytest -q` or with coverage `uv run pytest --cov=sieves`
- Docs (local): `uv run mkdocs serve`

## Extension Points

Adding a new predictive task:
- Create module under `sieves/tasks/predictive/<task_name>/`
- Implement `core.py` subclassing `PredictiveTask` and a `Bridge` for supported engines
- Provide prompt signature(s), prompt template(s), extraction, consolidation, and integration logic
- Add conversion to/from HF datasets if applicable

Adding a new engine:
- Create `sieves/engines/<engine_name>_.py`
- Subclass appropriate engine base (e.g., `PydanticEngine` or implement `InternalEngine` contract)
- Implement `build_executable(...)`, advertise `inference_modes`, and map engine types in `EngineType`
- Ensure serialization via `serialize()/deserialize()` works with `Config`

Pre/Post‑processing:
- Add chunkers or Ingestion connectors under `sieves/tasks/preprocessing/...`
- Extend distillation frameworks under `sieves/tasks/postprocessing/distillation/...`

## Prompts & Signatures

- Prompt templates are Jinja2; see each task’s default template and bridge logic
- Provide `fewshot_examples` as `pydantic` models matching the prompt signature
- For engines without strong schema guarantees, rely on task bridges to parse/validate outputs
- Use `strict_mode` when you want failures to surface; otherwise Nones are emitted on parse issues

## Caching & Performance

- `Pipeline(use_cache=True)` caches by `hash(doc.text or doc.uri)`
- Chunking reduces context length; engines may batch per `_batch_size` (−1 means "all")
- Prefer streaming/generator patterns for large corpora; tasks accept `Iterable[Doc]`

## Observability & Serialization

- Logging via `loguru` during pipeline execution
- Persist pipelines with `Pipeline.dump()` and reload via `Pipeline.load(path, task_kwargs)`
- Engine and task configs serialize through `sieves.serialization.Config` and `Serializable`

## Guardrails (For Agents)

Do:
- Adhere to typing and lint rules; run mypy/ruff/black before proposing changes
- Keep patches minimal; avoid unrelated refactors
- Respect optional dependencies; gate engine‑specific imports behind extras
- Update docs in `docs/` if you add public features (but avoid large doc rewrites without guidance)

Don’t:
- Commit secrets or modify CI/secrets; never embed API keys
- Perform destructive ops (e.g., mass renames, file deletions) without explicit instruction
- Change public APIs casually; ensure backward compatibility or document breaking changes

Escalate:
- Network calls to external services or model downloads
- Installing system packages for Ingestion/vision/accelerators
- Large dependency changes (adding/removing major engines)

## Verification Checklist

Before opening a PR or proposing a patch, ensure:
- Build/install succeeds: `uv sync [--extras]` and `uv run python -c "import sieves"`
- Lint passes: `uv run ruff check .` and formatting with black
- Types pass: `uv run mypy sieves`
- Tests pass: `uv run pytest` (include slow tests only if requested)
- New code covered by tests where practical
- Docs updated for new public APIs/features

## Secrets & Credentials

- Use environment variables only; never hardcode or commit creds
- Prefer provider‑agnostic clients (e.g., LiteLLM) where possible
- For local LLMs (Ollama), ensure service is reachable and models are pulled out‑of‑band

## CI & Releases

- GitHub Actions runs tests on PRs (see status badge in README)
- PyPI packaging metadata lives in `pyproject.toml`
- Keep versioning dynamic (setuptools‑scm) and changelog via PRs/commits

## Known Constraints

- Some engines do not support batching or few‑shotting uniformly; bridges handle compatibility
- Optional extras gate heavy deps (transformers, accelerate, Ingestion stacks, distillation)
- vLLM support exists but may be disabled/commented; verify environment before use

## Useful References

- Code docs: `docs/` (mkdocs) and module docstrings
- Engine guides: `docs/engines/*.md`
- Task guides: `docs/tasks/**/*.md`
- Getting started: `docs/guides/getting_started.md`

---

Maintainers: please update this file when:
- Adding/removing engines or tasks
- Changing installation or extras
- Updating coding standards or CI
- Introducing new workflows or guardrails
