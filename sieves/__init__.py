import sieves.tasks as tasks
from sieves.data import Doc

from .pipeline import Pipeline

__all__ = ["Doc", "tasks", "Pipeline"]

# todo
#  - Add predictive task
#  - Add DSPy engine
#  - Add GliX engine
#  - Add few-shot example support
#  - Allow for customization of prompt templates
