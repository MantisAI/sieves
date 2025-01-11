from __future__ import annotations

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

    @classmethod
    def get_engine_type(cls, engine: Engine[PromptSignature, Result, Model, InferenceMode]) -> EngineType:
        """Get engine type for self._engine.
        :param engine: Engine to check.
        :returns: Engine type for self._engine.
        :raises: ValueError if engine class not found in EngineType.
        """
        for et in EngineType:
            if isinstance(engine, et.value):
                return et
        raise ValueError(f"Engine class {engine.__class__.__name__} not found in EngineType.")


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
