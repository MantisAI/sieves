import enum

from . import dspy_engine, outlines_engine
from .core import Engine, InferenceMode, Model, PromptSignature, Result


class EngineType(enum.Enum):
    outlines = outlines_engine.Outlines
    dspy = dspy_engine.DSPy
    # jsonformer = 2


__all__ = [
    "dspy_engine",
    "Engine",
    "EngineType",
    "Result",
    "Model",
    "outlines_engine",
    "InferenceMode",
    "PromptSignature",
]
