"""Engine type enum and utilities."""

from __future__ import annotations

import enum

from sieves.engines.core import Engine, EngineInferenceMode, EngineModel, EnginePromptSignature, EngineResult
from sieves.engines.dspy_ import DSPy
from sieves.engines.gliner_ import GliNER
from sieves.engines.huggingface_ import HuggingFace
from sieves.engines.langchain_ import LangChain
from sieves.engines.outlines_ import Outlines


class ModelType(enum.Enum):
    """Available engine types."""

    dspy = DSPy
    gliner = GliNER
    huggingface = HuggingFace
    langchain = LangChain
    outlines = Outlines

    @classmethod
    def all(cls) -> tuple[ModelType, ...]:
        """Return all available engine types.

        :return tuple[ModelType, ...]: All available engine types.
        """
        return tuple(ModelType)

    @classmethod
    def get_engine_type(
        cls, engine: Engine[EnginePromptSignature, EngineResult, EngineModel, EngineInferenceMode]
    ) -> ModelType:
        """Return model type for specified engine.

        :param engine: Engine to get type for.
        :return ModelType: Engine type for self._engine.
        :raises ValueError: if engine class not found in ModelType.
        """
        for et in ModelType:
            if isinstance(engine, et.value):
                return et
        raise ValueError(f"Engine class {engine.__class__.__name__} not found in ModelType.")
