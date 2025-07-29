from __future__ import annotations

import enum

from .core import EngineInferenceMode, EngineModel, EnginePromptSignature, EngineResult, InternalEngine
from .engine_import import dspy_, glix_, huggingface_, instructor_, langchain_, ollama_, outlines_
from .missing import MissingEngine


class EngineType(enum.Enum):
    dspy = dspy_.DSPy
    glix = glix_.GliX
    huggingface = huggingface_.HuggingFace
    instructor = instructor_.Instructor
    langchain = langchain_.LangChain
    ollama = ollama_.Ollama
    outlines = outlines_.Outlines
    vllm = MissingEngine
    # vllm = vllm_.VLLM

    @classmethod
    def all(cls) -> tuple[EngineType, ...]:
        """Returns all available engine types.
        :return tuple[EngineType, ...]: All available engine types.
        """
        return tuple(engine_type for engine_type in EngineType if engine_type != EngineType.vllm)

    @classmethod
    def get_engine_type(
        cls, engine: InternalEngine[EnginePromptSignature, EngineResult, EngineModel, EngineInferenceMode]
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
