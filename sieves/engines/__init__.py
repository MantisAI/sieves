from __future__ import annotations

import enum

from . import dspy_, glix_, huggingface_, instructor_, langchain_, ollama_, outlines_
from .core import Engine, EngineInferenceMode, EngineModel, EnginePromptSignature, EngineResult
from .dspy_ import DSPy
from .glix_ import GliX
from .huggingface_ import HuggingFace
from .instructor_ import Instructor
from .langchain_ import LangChain
from .ollama_ import Ollama
from .outlines_ import Outlines


class EngineType(enum.Enum):
    dspy = dspy_.DSPy
    glix = glix_.GliX
    huggingface = huggingface_.HuggingFace
    instructor = instructor_.Instructor
    langchain = langchain_.LangChain
    ollama = ollama_.Ollama
    outlines = outlines_.Outlines

    @classmethod
    def all(cls) -> tuple[EngineType, ...]:
        """Returns all available engine types.
        :return tuple[EngineType, ...]: All available engine types.
        """
        return tuple(engine_type for engine_type in EngineType)

    @classmethod
    def get_engine_type(
        cls, engine: Engine[EnginePromptSignature, EngineResult, EngineModel, EngineInferenceMode]
    ) -> EngineType:
        """Returns engine type for specified engine.
        :param engine: Engine to get type for.
        :return EngineType: Engine type for self._engine.
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
    "langchain_",
    "LangChain",
    "huggingface_",
    "HuggingFace",
    "instructor_",
    "Instructor",
    "EngineResult",
    "EngineModel",
    "ollama_",
    "Ollama",
    "outlines_",
    "Outlines",
    "EngineInferenceMode",
    "EnginePromptSignature",
]
