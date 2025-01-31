from __future__ import annotations

import enum

from . import dspy_, glix_, huggingface_, langchain_, ollama_, outlines_
from .core import Engine, EngineInferenceMode, EngineModel, EnginePromptSignature, EngineResult
from .dspy_ import DSPy
from .glix_ import GliX
from .huggingface_ import HuggingFace
from .langchain_ import LangChain
from .ollama_ import Ollama
from .outlines_ import Outlines


class EngineType(enum.Enum):
    outlines = outlines_.Outlines
    dspy = dspy_.DSPy
    huggingface = huggingface_.HuggingFace
    glix = glix_.GliX
    ollama = ollama_.Ollama
    langchain = langchain_.LangChain

    @classmethod
    def all(cls) -> tuple[EngineType, ...]:
        """Returns all available engine types.
        :return tuple[EngineType, ...]: All available engine types.
        """
        return cls.outlines, cls.dspy, cls.huggingface, cls.glix, cls.ollama, cls.langchain

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
    "EngineResult",
    "EngineModel",
    "ollama_",
    "Ollama",
    "outlines_",
    "Outlines",
    "EngineInferenceMode",
    "EnginePromptSignature",
]
