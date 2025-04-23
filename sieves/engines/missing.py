import enum
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias

import pydantic

from sieves.engines.core import InternalEngine

PromptSignature: TypeAlias = Any
Model: TypeAlias = Any
Result: TypeAlias = Any


class InferenceMode(enum.Enum):
    any = Any


class MissingEngine(InternalEngine[PromptSignature, Result, Model, InferenceMode]):
    """Placeholder for engine that couldn't be imported due to missing dependencies."""

    @property
    def supports_few_shotting(self) -> bool:
        raise NotImplementedError

    @property
    def inference_modes(self) -> type[InferenceMode]:
        raise NotImplementedError

    def build_executable(
        self,
        inference_mode: InferenceMode,
        prompt_template: str | None,
        prompt_signature: type[PromptSignature] | PromptSignature,
        fewshot_examples: Iterable[pydantic.BaseModel] = (),
    ) -> Callable[[Iterable[dict[str, Any]]], Iterable[Result | None]]:
        raise NotImplementedError


DSPy = MissingEngine
GliX = MissingEngine
HuggingFace = MissingEngine
Instructor = MissingEngine
LangChain = MissingEngine
Ollama = MissingEngine
Outlines = MissingEngine
VLLM = MissingEngine
