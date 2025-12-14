# Preprocessing Documents

`sieves` provides several preprocessing tasks to prepare your documents for downstream processing. These tasks handle common operations like:
- Parsing various document formats (PDF, DOCX, etc.)
- Chunking long documents into manageable pieces
- Cleaning and standardizing text

## Document Parsing

Note: Ingestion libraries are optional and not installed by default. To use document ingestion, install them manually or install the `ingestion` extra:

```bash
pip install "sieves[ingestion]"
```

You can also install individual libraries directly (e.g., `pip install docling`).

### Using Ingestion

The `Ingestion` task uses the [docling](https://github.com/DS4SD/docling) or alternatively the [marker](https://github.com/VikParuchuri/marker) libraries to parse various document formats:

```python title="Basic document ingestion"
--8<-- "sieves/tests/docs/test_preprocessing.py:ingestion-basic"
```

It is possible to choose a specific output format between the supported (Markdown, HTML, JSON) and pass custom Docling or Marker converters in the `converter` parameter:

```python title="Custom converter with export format"
--8<-- "sieves/tests/docs/test_preprocessing.py:ingestion-custom-converter"
```

## Document Chunking

Long documents often need to be split into smaller chunks for processing by language models. `sieves` provides two chunking options:

### Using Chunking

The `Chunking` task uses the [chonkie](https://github.com/chonkie-ai/chonkie) library for intelligent document chunking:

```python title="Token-based chunking with Chonkie"
--8<-- "sieves/tests/docs/test_preprocessing.py:chunking-chonkie-basic"
```

## Combining Preprocessing Tasks

You can combine multiple preprocessing tasks in a pipeline. Here's an example that parses a PDF using the Ingestion task (using Docling as default) and then chunks it:

```python title="Combined preprocessing pipeline"
--8<-- "sieves/tests/docs/test_preprocessing.py:preprocessing-combined-pipeline"
```

## Customizing Preprocessing

### Progress

Progress bars are shown at the pipeline level. Tasks do not expose progress options.

### Metadata

Tasks can include metadata about their processing. Enable this with `include_meta`:

```python title="Enable metadata inclusion"
--8<-- "sieves/tests/docs/test_preprocessing.py:metadata-inclusion"
```

Access the metadata in the document's `meta` field:
```python title="Access preprocessing metadata"
--8<-- "sieves/tests/docs/test_preprocessing.py:metadata-access"
```

## Best Practices

1. **Document Size**: When working with large documents:
   - Always use chunking to break them into manageable pieces
   - Consider the chunk size based on your model's context window
   - Use appropriate chunk overlap to maintain context across chunks

2. **Error Handling**: When parsing documents:
   - Handle potential parsing errors gracefully
   - Verify that documents were parsed successfully before chunking
   - Check that the chunked text maintains document integrity

3. **Pipeline Order**: When combining tasks:
   - Always parse documents before chunking
   - Consider adding cleaning steps between parsing and chunking
   - Validate the output of each step before proceeding

4. **Text Cleaning**:
   - Choose cleaning functions based on your document types
   - Apply cleaning functions in a logical order (e.g., remove bullets before normalizing whitespace)
   - Test cleaning functions on sample documents to ensure they don't remove important content
