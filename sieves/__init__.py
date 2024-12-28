import sieves.tasks as tasks
from sieves.data import Doc

from .pipeline import Pipeline

__all__ = ["Doc", "tasks", "Pipeline"]

# todo
#  - Add predictive task
#  - Add more engines (DSPy, GliX (not generative!), JsonFormer)
#  - Add few-shot example support
#  - Chunking support (on task-level)
#  - Allow for customization of prompt templates
#  - Allow engine-specific settings in engine constructors
