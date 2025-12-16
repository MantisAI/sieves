# Documentation Testing Guide

This guide explains how to keep documentation examples tested and up-to-date using snippet injection.

---

## Quick Reference

### TL;DR

1. **Write tests** in `sieves/tests/examples/` with snippet markers
2. **Reference** them in docs using `--8<--` syntax
3. **Test** with `pytest sieves/tests/examples/`
4. **Build docs** with `mkdocs serve` or `mkdocs build`

### Snippet Markers in Python

```python
def test_my_example():
    """Test description."""
    # --8<-- [start:snippet-name]
    from sieves import Doc
    doc = Doc(text="Hello")
    # --8<-- [end:snippet-name]

    # Test code (not in docs)
    assert doc.text == "Hello"
```

### Reference in Markdown

```markdown
\```python title="Optional title"
--8<-- "sieves/tests/examples/test_my_file.py:snippet-name"
\```
```

### Common Commands

```bash
# Test examples only
uv run pytest sieves/tests/examples/ -v

# Build docs locally
uv run mkdocs serve   # http://127.0.0.1:8000

# Build for production
uv run mkdocs build --strict
```

---

## Overview

We use **snippet injection** to maintain a single source of truth for code examples:
- ✅ Write examples as actual **pytest tests** in `sieves/tests/examples/`
- ✅ Reference them in **markdown docs** using snippet markers
- ✅ Tests ensure examples stay **working and up-to-date**
- ✅ **DRY principle**: No duplicate code between tests and docs

## How It Works

### 1. Write Test Files with Snippet Markers

Create test files in `sieves/tests/examples/` with special markers:

```python
# sieves/tests/examples/test_my_feature.py

def test_basic_example():
    """Test demonstrating basic feature usage."""
    # --8<-- [start:basic-example]
    from sieves import Doc

    # Create a document
    doc = Doc(text="Hello world")
    print(doc.text)
    # --8<-- [end:basic-example]

    # Test assertions (not shown in docs)
    assert doc.text == "Hello world"
```

**Key points:**
- Markers: `# --8<-- [start:name]` and `# --8<-- [end:name]`
- Everything between markers appears in docs
- Code outside markers (like assertions) stays hidden
- Tests run normally with `pytest`

### 2. Reference Snippets in Documentation

In your markdown files (e.g., `docs/guides/my_guide.md`):

```markdown
# My Guide

Here's how to create a document:

\```python title="Creating a document"
--8<-- "sieves/tests/examples/test_my_feature.py:basic-example"
\```
```

The snippet path format is: `"path/to/file.py:snippet-name"`

### 3. Build and Verify

```bash
# Test examples
uv run pytest sieves/tests/examples/ -v

# Build docs locally to verify snippets render correctly
uv run mkdocs serve

# Check docs build (CI)
uv run mkdocs build
```

## Directory Structure

```
sieves/
├── tests/
│   ├── examples/           # ← Documentation example tests
│   │   ├── __init__.py
│   │   ├── test_getting_started.py
│   │   ├── test_classification.py
│   │   ├── test_preprocessing.py
│   │   └── ...
│   ├── tasks/              # ← Regular unit/integration tests
│   └── README.md           # ← This file
└── docs/
    ├── guides/
    │   ├── getting_started.md    # References snippets
    │   └── ...
    └── ...
```

## Naming Conventions

### Test Files
- `test_<guide_name>.py` for guide examples
- `test_<feature>_examples.py` for feature-specific examples

### Snippet Names
Use descriptive, kebab-case names:
- `basic-example`
- `doc-from-text`
- `advanced-pipeline`
- `label-descriptions`

## Best Practices

### 1. Keep Snippets Focused

```python
# ✅ Good: Focused example
def test_doc_creation():
    # --8<-- [start:doc-from-text]
    from sieves import Doc
    doc = Doc(text="Hello world")
    # --8<-- [end:doc-from-text]
    assert doc.text == "Hello world"

# ❌ Bad: Too much in one snippet
def test_everything():
    # --8<-- [start:huge-example]
    # 50 lines of code...
    # --8<-- [end:huge-example]
```

### 2. Hide Test Assertions

```python
def test_classification():
    # --8<-- [start:classify]
    from sieves import tasks
    result = tasks.Classification(labels=["a", "b"], model=model)
    # --8<-- [end:classify]

    # Assertions stay outside snippet markers
    assert result is not None
    assert len(result.labels) == 2
```

### 3. Use Skip Markers for Expensive Tests

```python
@pytest.mark.skip(reason="Requires API key and is expensive")
def test_api_example():
    # --8<-- [start:api-call]
    import os
    import dspy

    model = dspy.LM("claude-3-haiku-20240307", api_key=os.environ["ANTHROPIC_API_KEY"])
    # ... rest of example
    # --8<-- [end:api-call]

    # This test won't run in CI but snippet still appears in docs
```

### 4. Group Related Examples

```python
# test_classification_examples.py

def test_basic_classification():
    # --8<-- [start:basic]
    # ... basic example
    # --8<-- [end:basic]

def test_multi_label_classification():
    # --8<-- [start:multi-label]
    # ... multi-label example
    # --8<-- [end:multi-label]

def test_with_descriptions():
    # --8<-- [start:with-descriptions]
    # ... example with label descriptions
    # --8<-- [end:with-descriptions]
```

### 5. Add Context in Docstrings

```python
def test_advanced_pipeline():
    """
    Test advanced pipeline combining ingestion, chunking, and classification.

    This example demonstrates:
    - PDF parsing with Docling
    - Token-based chunking with Chonkie
    - Multi-label classification

    Referenced in: docs/guides/getting_started.md
    """
    # --8<-- [start:advanced-pipeline]
    # ... example code
    # --8<-- [end:advanced-pipeline]
```

## Markdown Usage Tips

### Basic Usage

```markdown
\```python
--8<-- "sieves/tests/examples/test_getting_started.py:basic-example"
\```
```

### With Title

```markdown
\```python title="Creating documents from text"
--8<-- "sieves/tests/examples/test_getting_started.py:doc-from-text"
\```
```

### Multiple Snippets in Sequence

```markdown
First, create a document:

\```python
--8<-- "sieves/tests/examples/test_doc.py:create-doc"
\```

Then process it:

\```python
--8<-- "sieves/tests/examples/test_doc.py:process-doc"
\```
```

## Testing Strategy

### Local Development

```bash
# Run only example tests
uv run pytest sieves/tests/examples/ -v

# Run specific test file
uv run pytest sieves/tests/examples/test_getting_started.py -v

# Run with output visible
uv run pytest sieves/tests/examples/ -v -s
```

### CI Integration

Add to your CI workflow:

```yaml
# .github/workflows/test.yml
- name: Test documentation examples
  run: uv run pytest sieves/tests/examples/ -v

- name: Build documentation
  run: uv run mkdocs build --strict
```

The `--strict` flag ensures MkDocs fails if snippets are missing or incorrect.

## Common Issues

### Issue: Snippet Not Found

**Error:** `Snippet 'my-example' not found in file`

**Solution:**
1. Check snippet name matches exactly (case-sensitive)
2. Verify file path is correct relative to repo root
3. Ensure markers use correct syntax: `# --8<-- [start:name]`

### Issue: Indentation Problems

**Error:** Code appears incorrectly indented in docs

**Solution:** The snippet extension handles this automatically with `dedent_subsections: true` in mkdocs.yml. If issues persist, check that your snippet starts at consistent indentation.

### Issue: Test Fails but Snippet Works in Docs

**Problem:** Assertions outside snippet fail, but snippet code is valid

**Solution:** This is expected! The snippet shows only working code. Fix the test assertions separately or mark test with `@pytest.mark.skip()` if it requires external resources.

## Migration Guide

### Converting Existing Docs

1. **Identify candidate examples** in your current documentation
2. **Create test file** for related examples
3. **Copy example code** into test function with markers
4. **Add assertions** to verify it works
5. **Replace markdown code** with snippet reference
6. **Test** locally with `mkdocs serve`

Example conversion:

**Before (in docs/guides/my_guide.md):**
```markdown
\```python
from sieves import Doc
doc = Doc(text="Hello")
\```
```

**After:**

1. Create `sieves/tests/examples/test_my_guide.py`:
```python
def test_basic_doc():
    # --8<-- [start:basic-doc]
    from sieves import Doc
    doc = Doc(text="Hello")
    # --8<-- [end:basic-doc]
    assert doc.text == "Hello"
```

2. Update `docs/guides/my_guide.md`:
```markdown
\```python
--8<-- "sieves/tests/examples/test_my_guide.py:basic-doc"
\```
```

## Configuration Reference

### mkdocs.yml

```yaml
markdown_extensions:
  - pymdownx.snippets:
      base_path: ['.']          # Search from repo root
      check_paths: true         # Fail if snippet path invalid
      dedent_subsections: true  # Auto-dedent code blocks
```

## Further Reading

- [pymdownx.snippets documentation](https://facelessuser.github.io/pymdown-extensions/extensions/snippets/)
- See existing examples in `sieves/tests/examples/test_getting_started.py`
- Open an issue if something isn't working

---

**Remember:** The goal is to make documentation trustworthy. If it's in the docs, it should work!
