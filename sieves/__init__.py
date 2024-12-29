import sieves.tasks as tasks
from sieves.data import Doc

from .pipeline import Pipeline

__all__ = ["Doc", "tasks", "Pipeline"]

# todo
#  - Add progress bar support in PredictiveTask (should work with Iterable returned by engines)
#  - Add more engines (GliX (not generative!), JsonFormer)
#  - Add few-shot example support
#    - https://dspy.ai/deep-dive/data-handling/examples/
#  - Chunking support (on task-level)
#  - Allow for customization of prompt templates
#  - Allow engine-specific settings in engine constructors
#  - Serialization
#  - Tasks
#    - Classification: add multi-class support
#    - NER
#    - Entity linking
#    - KG construction
#    - Translation
#    ...
