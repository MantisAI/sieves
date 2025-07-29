# Saving and Loading

`sieves` provides functionality to save your pipeline configurations to disk and load them later. This is useful for:

- Sharing pipeline configurations with others
- Versioning your pipelines
- Deploying pipelines to production

## Basic Pipeline Serialization

Here's a simple example of saving and loading a classification pipeline:

```python
import outlines
from sieves import Pipeline, Engine, tasks, Doc
from pathlib import Path

# Create a basic classification pipeline
model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
engine = Engine(model=outlines.models.transformers(model_name))
classifier = tasks.predictive.Classification(
    labels=["science", "politics"], 
    engine=engine
)
pipeline = Pipeline([classifier])

# Save the pipeline configuration
config_path = Path("classification_pipeline.yml")
pipeline.dump(config_path)

# Load the pipeline configuration
loaded_pipeline = Pipeline.load(
    config_path,
    [{"engine": {"model": outlines.models.transformers(model_name)}}]
)

# Use the loaded pipeline
doc = Doc(text="Special relativity applies to all physical phenomena in the absence of gravity.")
results = list(loaded_pipeline([doc]))
print(results[0].results["Classification"])
```

## Dealing with complex third-party objects

`sieves` doesn't serialize complex third-party objects. When loading pipelines, you need to provide initialization parameters for each task when loading:

```python
import chonkie
import tokenizers
import outlines
import pydantic
from sieves import Pipeline, Engine, tasks

# Create a tokenizer for chunking
tokenizer = tokenizers.Tokenizer.from_pretrained("bert-base-uncased")
chunker = tasks.preprocessing.Chonkie(
    chunker=chonkie.TokenChunker(tokenizer, chunk_size=512, chunk_overlap=50)
)

# Create an information extraction task
engine = Engine(model=outlines.models.transformers("HuggingFaceTB/SmolLM-135M-Instruct"))
class PersonInfo(pydantic.BaseModel):
    name: str
    age: int | None = None
    occupation: str | None = None

extractor = tasks.predictive.InformationExtraction(
    entity_type=PersonInfo,
    engine=engine
)

# Create and save the pipeline
pipeline = Pipeline([chunker, extractor])
pipeline.dump("extraction_pipeline.yml")

# Load the pipeline with initialization parameters for each task
loaded_pipeline = Pipeline.load(
    "extraction_pipeline.yml",
    [
        # Parameters for the chunker
        {"tokenizer": tokenizers.Tokenizer.from_pretrained("bert-base-uncased"),},
        # Parameters for the extractor
        {"engine": {"model": outlines.models.transformers("HuggingFaceTB/SmolLM-135M-Instruct")}}
    ]
)
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

      When loading pipelines, provide all required initialization parameters and ensure you're loading a pipeline with a compatible `sieves` version.

!!! warning Limitations

      - Model weights are not saved in the configuration files
      - Complex third-party objects (everything beyond primitives or collections thereof) may not be serializable
      - API keys and credentials must be managed separately
