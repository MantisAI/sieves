"""Unified engine wrapper that dispatches to a concrete backend based on model type."""

from __future__ import annotations

from collections.abc import Iterable
from typing import override

import pydantic

from sieves.engines.core import (
    EngineInferenceMode,
    EngineModel,
    EnginePromptSignature,
    EngineResult,
    Executable,
    InternalEngine,
)
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
from sieves.engines.engine_type import EngineType
from sieves.engines.utils import GenerationSettings

PromptSignature = (
    dspy_.PromptSignature
    | glix_.PromptSignature
    | huggingface_.PromptSignature
    | instructor_.PromptSignature
    | langchain_.PromptSignature
    | ollama_.PromptSignature
    | outlines_.PromptSignature
    | vllm_.PromptSignature
)
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
Result = (
    dspy_.Result
    | glix_.Result
    | huggingface_.Result
    | instructor_.Result
    | langchain_.Result
    | ollama_.Result
    | outlines_.Result
    | vllm_.Result
)
InferenceMode = (
    dspy_.InferenceMode
    | glix_.InferenceMode
    | huggingface_.InferenceMode
    | instructor_.InferenceMode
    | langchain_.InferenceMode
    | ollama_.InferenceMode
    | outlines_.InferenceMode
    | vllm_.InferenceMode
)


class Engine(InternalEngine[PromptSignature, Result, Model, InferenceMode]):
    """Facade over specific engines (Outlines, Instructor, LangChain, etc.)."""

    def __init__(
        self,
        model: Model | None = None,
        generation_settings: GenerationSettings | None = None,
    ):
        """Initialize new engine.

        :param model: Model to run. If None, a default model (HuggingFaceTB/SmolLM-360M-Instruct with Outlines) is used.
        :param generation_settings: Settings for structured generation.
        """
        gen_settings = generation_settings or GenerationSettings()
        super().__init__(model or Engine._init_default_model(), gen_settings)
        self._engine: InternalEngine[PromptSignature, Result, Model, InferenceMode] = self._init_engine()

    @classmethod
    def _init_default_model(cls) -> Model:  # noqa: D401
        """Initialize default model (HuggingFaceTB/SmolLM-360M-Instruct with Outlines).

        :return: Initialized default model.
        """
        import outlines
        import transformers

        model_name = "HuggingFaceTB/SmolLM-360M-Instruct"

        return outlines.models.from_transformers(
            transformers.AutoModelForCausalLM.from_pretrained(model_name),
            transformers.AutoTokenizer.from_pretrained(model_name),
        )

    def _init_engine(self) -> InternalEngine[EnginePromptSignature, EngineResult, EngineModel, EngineInferenceMode]:  # noqa: D401
        """Initialize internal engine object.

        :return Engine: Engine.
        :raises: ValueError if model type isn't supported.
        """
        model_type = type(self._model)
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
                    model=self._model,
                    generation_settings=self._generation_settings,
                )
                assert isinstance(internal_engine, InternalEngine)

                return internal_engine

        raise ValueError(
            f"Model type {self.model.__class__} is not supported. Please check the documentation and ensure you're "
            f"providing a supported model type."
        )

    @property
    def supports_few_shotting(self) -> bool:
        """Whether wrapped engine supports few-shotting."""
        return self._engine.supports_few_shotting

    @property
    def inference_modes(self) -> type[InferenceMode]:
        """Supported inference modes of wrapped engine."""
        return self._engine.inference_modes

    @override
    def build_executable(
        self,
        inference_mode: InferenceMode,
        prompt_template: str | None,
        prompt_signature: type[PromptSignature] | PromptSignature,
        fewshot_examples: Iterable[pydantic.BaseModel] = (),
    ) -> Executable[Result | None]:
        return self._engine.build_executable(
            inference_mode=inference_mode,
            prompt_template=prompt_template,
            prompt_signature=prompt_signature,
            fewshot_examples=fewshot_examples,
        )

    def get_engine_type(self) -> EngineType:  # noqa: D401
        """Return engine type for specified engine.

        :return EngineType: Engine type for self._engine.
        :raises: ValueError if engine class not found in EngineType.
        """
        return EngineType.get_engine_type(self._engine)
