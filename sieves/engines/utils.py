"""Utils for engines."""

import outlines
import transformers

from sieves.engines.core import Engine, EngineInferenceMode, EngineModel, EnginePromptSignature, EngineResult
from sieves.engines.engine_import import (
    dspy_,
    glix_,
    huggingface_,
    instructor_,
    langchain_,
    ollama_,
    outlines_,
    vllm_,
)
from sieves.engines.types import GenerationSettings

Model = (
    dspy_.Model
    | glix_.Model
    | huggingface_.Model
    | instructor_.Model
    | langchain_.Model
    | ollama_.Model
    | outlines_.Model
    | vllm_.Model
)


def init_default_model() -> outlines.models.Transformers:  # noqa: D401
    """Initialize default model (HuggingFaceTB/SmolLM-360M-Instruct with Outlines).

    :return: Initialized default model.
    """
    model_name = "HuggingFaceTB/SmolLM-360M-Instruct"

    return outlines.models.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained(model_name),
        transformers.AutoTokenizer.from_pretrained(model_name),
    )


def init_engine(
    model: Model, generation_settings: GenerationSettings
) -> Engine[EnginePromptSignature, EngineResult, EngineModel, EngineInferenceMode]:  # noqa: D401
    """Initialize internal engine object.

    :param model: Model to use.
    :param generation_settings: Settings for structured generation.
    :return Engine: Engine.
    :raises ValueError: If model type isn't supported.
    """
    model_type = type(model)
    module_engine_map = {
        dspy_: dspy_.DSPy,
        glix_: glix_.GliX,
        huggingface_: huggingface_.HuggingFace,
        instructor_: instructor_.Instructor,
        langchain_: langchain_.LangChain,
        ollama_: ollama_.Ollama,
        outlines_: outlines_.Outlines,
        # vllm_: vllm_.VLLM,
    }

    for module, engine_type in module_engine_map.items():
        try:
            module_model_types = module.Model.__args__
        except AttributeError:
            module_model_types = (module.Model,)

        if any(issubclass(model_type, module_model_type) for module_model_type in module_model_types):
            internal_engine = engine_type(
                model=model,
                generation_settings=generation_settings,
            )
            assert isinstance(internal_engine, Engine)

            return internal_engine

    raise ValueError(
        f"Model type {model.__class__} is not supported. Please check the documentation and ensure you're "
        f"providing a supported model type."
    )
