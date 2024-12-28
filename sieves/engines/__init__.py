import enum

from . import outlines_engine
from .core import Engine, InferenceGenerator, Model, PromptSignature, Result


class EngineType(enum.Enum):
    outlines = 0
    # dspy = 1
    # jsonformer = 2


__all__ = [
    "Engine",
    "EngineType",
    "Result",
    "Model",
    "outlines_engine",
    "InferenceGenerator",
    "PromptSignature",
]
