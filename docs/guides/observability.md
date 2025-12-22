# Observability and Usage Tracking

`sieves` provides built-in tools for monitoring your Document AI pipelines. By enabling metadata collection, you can inspect raw model responses and track token consumption for both local and remote models.

## The `meta` Field

Every `Doc` object in `sieves` contains a `meta` dictionary. When `include_meta=True` (which is the default for predictive tasks), this dictionary is populated with detailed execution traces.

### Raw Model Outputs

`sieves` captures the "raw" output from the underlying language model before it is parsed into your final structured format. This is invaluable for debugging prompt failures or investigating unexpected model behavior.

The raw outputs are stored in `doc.meta[task_id]['raw']`. Since documents can be split into multiple chunks, this field contains a list of raw responses—one for each chunk.

#### Example: Inspecting Raw Output

```python
from sieves.tasks import Classification

# The include_meta flag is True by default.
task = Classification(labels=["science", "politics"], model=model)
results = list(task(docs))

# Inspect raw model responses for the first document.
print(results[0].meta['Classification']['raw'])
```

**Example Result for DSPy:**
```python
[
    {
        'prompt': None,
        'messages': [...],
        'response': ModelResponse(...),
        'usage': {'prompt_tokens': 556, 'completion_tokens': 32, ...}
    }
]
```

**Example Result for Outlines (JSON mode):**
```python
['{
  "science": 0.95,
  "politics": 0.05
}']
```

---

## Token Usage Tracking

`sieves` automatically tracks input and output tokens across your pipeline. Token data is aggregated at three levels:

1.  **Per Chunk**: Detailed usage for every individual model call.
2.  **Per Task**: Aggregated usage for a specific task within a document.
3.  **Per Document**: Total running total of tokens consumed by a document across all tasks.

### Accessing Usage Data

Usage statistics are stored under the `usage` key in the metadata.

*   **Task-specific usage**: `doc.meta[task_id]['usage']`
*   **Total document usage**: `doc.meta['usage']`

#### Example Usage Structure

```python
# The total tokens consumed by this document across the entire pipeline.
total_usage = doc.meta['usage']
print(f"Total Input: {total_usage['input_tokens']}, Total Output: {total_usage['output_tokens']}")

# The detailed usage for a specific classification task.
task_meta = doc.meta['Classification']
print(f"Task Input: {task_meta['usage']['input_tokens']}")

# The per-chunk usage for the classification task.
for i, chunk_usage in enumerate(task_meta['usage']['chunks']):
    print(f"Chunk {i}: {chunk_usage['input_tokens']} in, {chunk_usage['output_tokens']} out")
```

---

## Native vs. Approximate Counting

`sieves` uses a multi-tiered approach to ensure you always have token data, even when model frameworks don't provide it natively.

### Native Tracking (DSPy & LangChain)
For backends like **DSPy** and **LangChain**, `sieves` extracts token counts directly from the model provider's metadata (e.g., OpenAI or Anthropic response headers). This is the most accurate form of tracking.

!!! note "DSPy Caching"
    DSPy's internal caching may return 0 or `None` for tokens if a result is retrieved from the local cache rather than the remote API.

### Approximate Estimation (Outlines, HuggingFace, GliNER)
For local models or frameworks that don't expose native counts, `sieves` uses the model's own **tokenizer** to estimate usage:
1.  **Input Tokens**: Counted by encoding the fully rendered prompt string.
2.  **Output Tokens**: Counted by encoding the raw generated output string.

If a local tokenizer is not available (e.g., when using a remote API via Outlines without a local weight clone), `sieves` will attempt to fall back to `tiktoken` (for OpenAI-compatible models) or return `None`.

```
