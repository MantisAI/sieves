"""Engine type enum and utilities."""

from __future__ import annotations

import enum

from .core import Engine, EngineInferenceMode, EngineModel, EnginePromptSignature, EngineResult
from .engine_import import dspy_, glix_, huggingface_, langchain_, outlines_


class EngineType(enum.Enum):
    """Available engine types."""

    dspy = dspy_.DSPy
    glix = glix_.GliX
    huggingface = huggingface_.HuggingFace
    langchain = langchain_.LangChain
    outlines = outlines_.Outlines

    @classmethod
    def all(cls) -> tuple[EngineType, ...]:
        """Return all available engine types.

        :return tuple[EngineType, ...]: All available engine types.
        """
        return tuple(EngineType)

    @classmethod
    def get_engine_type(
        cls, engine: Engine[EnginePromptSignature, EngineResult, EngineModel, EngineInferenceMode]
    ) -> EngineType:
        """Return engine type for specified engine.

        :param engine: Engine to get type for.
        :return EngineType: Engine type for self._engine.
        :raises ValueError: if engine class not found in EngineType.
        """
        for et in EngineType:
            if isinstance(engine, et.value):
                return et
        raise ValueError(f"Engine class {engine.__class__.__name__} not found in EngineType.")
