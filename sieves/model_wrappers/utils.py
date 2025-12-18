"""Utils for model wrappers."""

import outlines
import transformers

from sieves.model_wrappers import (
    dspy_,
    gliner_,
    huggingface_,
    langchain_,
    outlines_,
)
from sieves.model_wrappers.core import (
    ModelWrapper,
    ModelWrapperInferenceMode,
    ModelWrapperModel,
    ModelWrapperPromptSignature,
    ModelWrapperResult,
)
from sieves.model_wrappers.types import ModelSettings

Model = dspy_.Model | gliner_.Model | huggingface_.Model | langchain_.Model | outlines_.Model


def init_default_model() -> outlines.models.Transformers:  # noqa: D401
    """Initialize default model (HuggingFaceTB/SmolLM-360M-Instruct with Outlines).

    :return: Initialized default model.
    """
    model_name = "HuggingFaceTB/SmolLM-360M-Instruct"

    return outlines.models.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained(model_name),
        transformers.AutoTokenizer.from_pretrained(model_name),
    )


def init_model_wrapper(
    model: Model, model_settings: ModelSettings
) -> ModelWrapper[ModelWrapperPromptSignature, ModelWrapperResult, ModelWrapperModel, ModelWrapperInferenceMode]:  # noqa: D401
    """Initialize internal model wrapper object.

    :param model: Model to use.
    :param model_settings: Settings for structured generation.
    :return ModelWrapper: ModelWrapper.
    :raises ValueError: If model type isn't supported.
    """
    model_type = type(model)
    module_wrapper_map = {
        dspy_: getattr(dspy_, "DSPy", None),
        gliner_: getattr(gliner_, "GliNER", None),
        huggingface_: getattr(huggingface_, "HuggingFace", None),
        langchain_: getattr(langchain_, "LangChain", None),
        outlines_: getattr(outlines_, "Outlines", None),
    }

    for module, model_wrapper_type in module_wrapper_map.items():
        assert hasattr(module, "Model")
        assert model_wrapper_type

        try:
            module_model_types = module.Model.__args__
        except AttributeError:
            module_model_types = (module.Model,)

        if any(issubclass(model_type, module_model_type) for module_model_type in module_model_types):
            internal_model_wrapper = model_wrapper_type(
                model=model,
                model_settings=model_settings,
            )
            assert isinstance(internal_model_wrapper, ModelWrapper)

            return internal_model_wrapper

    raise ValueError(
        f"Model type {model.__class__} is not supported. Please check the documentation and ensure that (1) you're "
        f"providing a supported model type and that (2) the corresponding library is installed in your environment."
    )
