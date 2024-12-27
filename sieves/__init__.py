import sieves.tasks as tasks
from sieves.data import Doc, Resource, chunkers

from .pipeline import Pipeline

__all__ = ["chunkers", "Doc", "Resource", "tasks", "Pipeline"]

# todo
#  - Rethink task architecture - tasks are coupled to engines (engine design, not ours). How to solve this elegantly?
#    -> common steps seem to be:
#       1. create prompt template (Jinja should be fine everywhere)
#       2. build prompt by injecting into template (different output formats!)
#       3. define output signature (Signature in DSPy, Pydantic objects in outlines, JSON schema in jsonformers)
#       4. build callable object (Predict in DSPy, Function in outlines, Jsonformer in jsonformers)
#       5. execute callable objects. parsing happens automatically
#    -> have Engine in Task, not the other way around, and have Engine class provide primitives and routines for
#       building those primitives?
#  - Add few-shot example support
