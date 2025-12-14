# Saving and Loading

`sieves` provides functionality to save your pipeline configurations to disk and load them later. This is useful for:

- Sharing pipeline configurations with others
- Versioning your pipelines
- Deploying pipelines to production

## Basic Pipeline Serialization

Here's a simple example of saving and loading a classification pipeline:

```python title="Basic pipeline serialization"
--8<-- "sieves/tests/docs/test_serialization.py:serialization-basic-pipeline"
```

## Dealing with complex third-party objects

`sieves` doesn't serialize complex third-party objects. When loading pipelines, you need to provide initialization parameters for each task when loading:

```python title="Complex pipeline serialization"
--8<-- "sieves/tests/docs/test_serialization.py:serialization-complex-pipeline"
```

## Understanding Pipeline Configuration Files

Pipeline configurations are saved as YAML files. Here's an example of what a configuration file looks like:

```yaml
cls_name: sieves.pipeline.core.Pipeline
version: 0.11.1
tasks:
  is_placeholder: false
  value:
    - cls_name: sieves.tasks.preprocessing.chunkers.Chunker
      tokenizer:
        is_placeholder: true
        value: tokenizers.Tokenizer
      chunk_size:
        is_placeholder: false
        value: 512
      chunk_overlap:
        is_placeholder: false
        value: 50
      task_id:
        is_placeholder: false
        value: Chunker
    - cls_name: sieves.tasks.predictive.information_extraction.core.InformationExtraction
      engine:
        is_placeholder: false
        value:
          cls_name: sieves.engines.outlines_.Outlines
          model:
            is_placeholder: true
            value: outlines.models.transformers
```

The configuration file contains:

- The full class path of the pipeline and its tasks
- Version information
- Task-specific parameters and their values
- Placeholders for components that need to be provided during loading

!!! info Parameter management

      When loading pipelines, provide all required initialization parameters (e.g. models) and ensure you're loading a pipeline with a compatible `sieves` version. `GenerationSettings` is optional unless you want to override defaults.

!!! warning Limitations

      - Model weights are not saved in the configuration files
      - Complex third-party objects (everything beyond primitives or collections thereof) may not be serializable
      - API keys and credentials must be managed separately
