# Docling

[Docling](https://github.com/DS4SD/docling) is a tool for document parsing and ingestion.

Note: This task depends on optional ingestion libraries, which are not installed by default. Install them via the ingestion extra, or install the library directly.

Examples:

```bash
pip install "sieves[ingestion]"   # installs ingestion deps via extra
# or install the library directly
pip install docling
```

## Usage

```python
--8<-- "sieves/tests/docs/test_preprocessing_usage.py:docling-usage"
```

---

::: sieves.tasks.preprocessing.ingestion.docling_
