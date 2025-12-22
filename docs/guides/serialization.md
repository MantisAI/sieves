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

`sieves` doesn't serialize complex third-party objects. When loading pipelines, you need to provide initialization parameters for tasks that use non-serializable components.

### Set Up Pipeline Components

First, create a pipeline with third-party objects like tokenizers and models:

```python title="Setting up components"
--8<-- "sieves/tests/docs/test_serialization.py:serialization-complex-setup"
```

### Define Entity Schema and Task

Define the schema for information extraction and create the task:

```python title="Entity and task definition"
--8<-- "sieves/tests/docs/test_serialization.py:serialization-complex-entity-task"
```

### Save the Pipeline

Create and save the pipeline configuration:

```python title="Saving the pipeline"
--8<-- "sieves/tests/docs/test_serialization.py:serialization-complex-save"
```

The YAML file stores the pipeline structure but includes placeholders for non-serializable objects.

### Load with Initialization Parameters

When loading, provide the actual objects for each task's placeholders:

```python title="Loading with init parameters"
--8<-- "sieves/tests/docs/test_serialization.py:serialization-complex-load"
```

The `init_params` list corresponds to each task in the pipeline, providing the values needed to reconstruct non-serializable components.

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
      model_wrapper:
        is_placeholder: false
        value:
          cls_name: sieves.model_wrappers.outlines_.Outlines
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

      When loading pipelines, provide all required initialization parameters (e.g. models) and ensure you're loading a pipeline with a compatible `sieves` version. `ModelSettings` is optional unless you want to override defaults.

!!! warning Limitations

      - Model weights are not saved in the configuration files
      - Complex third-party objects (everything beyond primitives or collections thereof) are not serializable
      - API keys and credentials must be managed separately

## Troubleshooting

### Common Issues

#### "Missing required placeholder" error during load

**Symptom**: `KeyError` or error about missing placeholders when loading a pipeline.

**Cause**: You didn't provide initialization parameters for all placeholders in the saved configuration.

**Solution**: Check the YAML file to see which parameters are marked as placeholders:

```python
# Read the config to see what placeholders exist.
import yaml
with open("pipeline.yml", "r") as f:
    config = yaml.safe_load(f)
    print(config)  # Look for "is_placeholder: true" entries.
```

Provide `init_params` for each task that has placeholders:

```python
loaded_pipeline = Pipeline.load(
    "pipeline.yml",
    [
        {"model": your_model},           # Task 0 placeholders.
        {"tokenizer": your_tokenizer},   # Task 1 placeholders.
    ]
)
```

#### Version compatibility warnings

**Symptom**: Warning about `sieves` version mismatch when loading pipelines.

**Cause**: The pipeline was saved with a different version of `sieves` than you're currently using.

**Impact**:

- Path version differences (0.11.1 vs. 0.11.2): Usually safe
- Major version differences (e.g. 0.22.0 vs. 1.0.1) and minor version differences with major < 1 (e.g. 0.11.x vs. 0.12.x): May have breaking changes

**Solution**:
```bash
# Install the version that was used to create the pipeline.
pip install sieves==0.11.1  # Match the version in the YAML.

# Or: Update the pipeline by re-saving it with the current version.
pipeline.dump("pipeline_updated.yml")
```

#### Serialization fails for custom objects

**Symptom**: Error when calling `pipeline.dump()` with custom tasks or parameters.

**Cause**: Custom objects that aren't primitive types (str, int, float, bool, list, dict) can't be automatically serialized.

**Solution**: Mark these as placeholders by ensuring they're provided during pipeline creation, then supply them again during load:

```python
# When creating the pipeline.
custom_task = MyCustomTask(complex_object=my_object)
pipeline = Pipeline([custom_task])
pipeline.dump("pipeline.yml")  # complex_object becomes a placeholder.

# When loading.
loaded = Pipeline.load("pipeline.yml", [{"complex_object": my_object}])
```

#### Model weights not loading

**Symptom**: Loaded pipeline doesn't have model weights.

**Cause**: `sieves` doesn't save model weights in configuration files (they're too large).

**Solution**: Always provide fresh model instances in `init_params`:

```python
# Load the model separately (weights will be downloaded/loaded).
model = outlines.models.transformers(
    "HuggingFaceTB/SmolLM-135M-Instruct"
)

# Then load the pipeline with the model.
loaded = Pipeline.load("pipeline.yml", [{"model": model}])
```

#### Task ID mismatches after loading

**Symptom**: Results are stored under different keys than expected.

**Cause**: Task IDs changed between save and load.

**Solution**: Specify explicit task IDs when creating tasks:

```python
# When creating.
classifier = tasks.Classification(
    labels=["science", "politics"],
    model=model,
    task_id="my_classifier"  # Explicit ID.
)

# The results will always be in doc.results["my_classifier"].
```

### Best Practices

1. **Version control configurations**: Store YAML files in git alongside code
2. **Document init_params**: Add comments explaining what placeholders need
3. **Test load immediately**: After saving, try loading to catch serialization issues
4. **Separate model loading**: Keep model initialization code separate from pipeline config
5. **Use version pinning**: Pin `sieves` version in requirements.txt for reproducibility

## Related Guides

- **[Custom Tasks](custom_tasks.md)** - Custom tasks often require init_params during load
- **[Task Optimization](optimization.md)** - Save optimized prompts in pipeline configs
- **[Task Distillation](distillation.md)** - Save distilled model paths in configurations
