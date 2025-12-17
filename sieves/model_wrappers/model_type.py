"""Model type enum and utilities."""

from __future__ import annotations

import enum

from sieves.model_wrappers.core import (
    ModelWrapper,
    ModelWrapperInferenceMode,
    ModelWrapperModel,
    ModelWrapperPromptSignature,
    ModelWrapperResult,
)
from sieves.model_wrappers.dspy_ import DSPy
from sieves.model_wrappers.gliner_ import GliNER
from sieves.model_wrappers.huggingface_ import HuggingFace
from sieves.model_wrappers.langchain_ import LangChain
from sieves.model_wrappers.outlines_ import Outlines


class ModelType(enum.Enum):
    """Available model types."""

    dspy = DSPy
    gliner = GliNER
    huggingface = HuggingFace
    langchain = LangChain
    outlines = Outlines

    @classmethod
    def all(cls) -> tuple[ModelType, ...]:
        """Return all available model types.

        :return tuple[ModelType, ...]: All available model types.
        """
        return tuple(ModelType)

    @classmethod
    def get_model_type(
        cls,
        model_wrapper: ModelWrapper[
            ModelWrapperPromptSignature, ModelWrapperResult, ModelWrapperModel, ModelWrapperInferenceMode
        ],
    ) -> ModelType:
        """Return model type for specified model wrapper.

        :param model_wrapper: ModelWrapper to get type for.
        :return ModelType: Model type for self._model_wrapper.
        :raises ValueError: if model wrapper class not found in ModelType.
        """
        for mt in ModelType:
            if isinstance(model_wrapper, mt.value):
                return mt
        raise ValueError(f"ModelWrapper class {model_wrapper.__class__.__name__} not found in ModelType.")
