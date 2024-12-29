import enum

from . import dspy_, outlines_
from .core import Engine, InferenceMode, Model, PromptSignature, Result


class EngineType(enum.Enum):
    outlines = outlines_.Outlines
    dspy = dspy_.DSPy
    # jsonformer = 2


__all__ = [
    "dspy_.py",
    "Engine",
    "EngineType",
    "Result",
    "Model",
    "outlines_.py",
    "InferenceMode",
    "PromptSignature",
]
