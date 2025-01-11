import enum

from . import dspy_, glix_, huggingface_, outlines_
from .core import Engine, InferenceMode, Model, PromptSignature, Result
from .dspy_ import DSPy
from .glix_ import GliX
from .huggingface_ import HuggingFace
from .outlines_ import Outlines


class EngineType(enum.Enum):
    outlines = outlines_.Outlines
    dspy = dspy_.DSPy
    huggingface = huggingface_.HuggingFace
    glix = glix_.GliX


__all__ = [
    "dspy_",
    "DSPy",
    "Engine",
    "EngineType",
    "glix_",
    "GliX",
    "huggingface_",
    "HuggingFace",
    "Result",
    "Model",
    "outlines_",
    "Outlines",
    "InferenceMode",
    "PromptSignature",
]
