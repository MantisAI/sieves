# Preprocessing Documents

`sieves` provides several preprocessing tasks to prepare your documents for downstream processing. These tasks handle common operations like:
- Parsing various document formats (PDF, DOCX, etc.)
- Chunking long documents into manageable pieces
- Cleaning and standardizing text

## Document Parsing

### Using Ingestion

The `Ingestion` task uses the [docling](https://github.com/DS4SD/docling) or alternatively the [marker](https://github.com/VikParuchuri/marker) libraries to parse various document formats:

```python
from sieves import Pipeline, tasks, Doc

# Create a document parser
parser = tasks.preprocessing.Ingestion()

# Create a pipeline with the parser
pipeline = Pipeline([parser])

# Process documents
docs = [
    Doc(uri="path/to/document.pdf"),
    Doc(uri="path/to/another.docx")
]
processed_docs = list(pipeline(docs))

# Access the parsed text
for doc in processed_docs:
    print(doc.text)
```

It is possible to choose a specific output format between the supported (Markdown, HTML, JSON) and pass custom Docling or Marker converters in the `converter` parameter:

```python
from sieves import Pipeline, tasks, Doc

from docling.document_converter import DocumentConverter

# Create a document parser
parser = tasks.preprocessing.Ingestion(converter=DocumentConverter(), export_format="html")

# Create a pipeline with the parser
pipeline = Pipeline([parser])

# Process documents
docs = [
    Doc(uri="path/to/document.pdf"),
    Doc(uri="path/to/another.docx")
]
processed_docs = list(pipeline(docs))

# Access the parsed text
for doc in processed_docs:
    print(doc.text)
```

### Using Unstructured

The `Unstructured` task uses the [unstructured](https://github.com/Unstructured-IO/unstructured/) library, which provides robust document parsing capabilities:

```python
from sieves import Pipeline, tasks, Doc
from unstructured.cleaners.core import (
    clean_extra_whitespace
)

# Create an unstructured parser with cleaning functions
parser = tasks.preprocessing.Unstructured(
    # Add cleaning functions to process the text
    cleaners=(clean_extra_whitespace,)  # Normalize whitespace
)

# Create and run the pipeline
pipeline = Pipeline([parser])
docs = [Doc(text="● This is a dummy   document.®")]
processed_docs = list(pipeline(docs))
```

### Text Cleaning Functions

The Unstructured library provides several cleaning functions that you can combine:

```python
from unstructured.cleaners.core import (
    clean,                    # Combined cleaning function
    clean_bullets,           # Remove bullet points (•, -, etc.)
    clean_ordered_bullets,   # Remove numbered bullets (1., a., etc.)
    clean_extra_whitespace,  # Normalize whitespace
    clean_dashes,           # Remove dashes
    clean_non_ascii_chars,  # Remove non-ASCII characters
    clean_trailing_punctuation,  # Remove trailing punctuation
    bytes_string_to_string  # Convert byte strings to normal strings
)
```

## Document Chunking

Long documents often need to be split into smaller chunks for processing by language models. `sieves` provides two chunking options:

### Using Chonkie

The `Chonkie` task uses the [chonkie](https://github.com/chonkie-ai/chonkie) library for intelligent document chunking:

```python
import chonkie
import tokenizers
from sieves import Pipeline, tasks, Doc

# Create a tokenizer for chunking
tokenizer = tokenizers.Tokenizer.from_pretrained("bert-base-uncased")

# Create a token-based chunker
chunker = tasks.preprocessing.Chonkie(
    chunker=chonkie.TokenChunker(tokenizer, chunk_size=512, chunk_overlap=50)
)

# Create and run the pipeline
pipeline = Pipeline([chunker])
doc = Doc(text="Your long document text here...")
chunked_docs = list(pipeline([doc]))

# Access the chunks
for chunk in chunked_docs[0].chunks:
    print(f"Chunk: {chunk}")
```

## Combining Preprocessing Tasks

You can combine multiple preprocessing tasks in a pipeline. Here's an example that parses a PDF using the Ingestion task (using Docling as default) and then chunks it with Chonkie:

```python
from sieves import Pipeline, tasks, Doc
import chonkie
import tokenizers

# Create the preprocessing tasks
parser = tasks.preprocessing.Ingestion()
tokenizer = tokenizers.Tokenizer.from_pretrained("bert-base-uncased")
chunker = tasks.preprocessing.Chonkie(
    chunker=chonkie.TokenChunker(tokenizer, chunk_size=512, chunk_overlap=50)
)

# Create a pipeline with both tasks
pipeline = Pipeline([parser, chunker])

# Process a document
doc = Doc(uri="path/to/document.pdf")
processed_doc = list(pipeline([doc]))[0]

# Access the chunks
print(f"Number of chunks: {len(processed_doc.chunks)}")
for i, chunk in enumerate(processed_doc.chunks):
    print(f"Chunk {i}: {chunk[:100]}...")  # Print first 100 chars of each chunk
```

## Customizing Preprocessing

### Progress

Progress bars are shown at the pipeline level. Tasks do not expose progress options.

### Metadata

Tasks can include metadata about their processing. Enable this with `include_meta`:

```python
parser = tasks.preprocessing.Ingestion(include_meta=True)
```

Access the metadata in the document's `meta` field:
```python
doc = processed_docs[0]
print(doc.meta["Ingestion"])  # Access parser metadata
print(doc.meta["Chunker"])  # Access chunker metadata
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
