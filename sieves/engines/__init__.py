import enum

from . import dspy_, huggingface_, outlines_
from .core import Engine, InferenceMode, Model, PromptSignature, Result


class EngineType(enum.Enum):
    outlines = outlines_.Outlines
    dspy = dspy_.DSPy
    huggingface = huggingface_.HuggingFace
    # jsonformer = 2


__all__ = [
    "dspy_",
    "Engine",
    "EngineType",
    "huggingface_",
    "Result",
    "Model",
    "outlines_",
    "InferenceMode",
    "PromptSignature",
]
