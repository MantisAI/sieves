"""Fallback engine types when optional dependencies are unavailable."""

import enum
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias, override

import pydantic

from sieves.engines.core import InternalEngine

PromptSignature: TypeAlias = Any
Model: TypeAlias = Any
Result: TypeAlias = Any


class InferenceMode(enum.Enum):
    """Placeholder mode for unsupported engines."""

    any = Any


class MissingEngine(InternalEngine[PromptSignature, Result, Model, InferenceMode]):
    """Placeholder for engine that couldn't be imported due to missing dependencies."""

    @override
    @property
    def supports_few_shotting(self) -> bool:
        raise NotImplementedError

    @override
    @property
    def inference_modes(self) -> type[InferenceMode]:
        raise NotImplementedError

    @override
    def build_executable(
        self,
        inference_mode: InferenceMode,
        prompt_template: str | None,
        prompt_signature: type[PromptSignature] | PromptSignature,
        fewshot_examples: Iterable[pydantic.BaseModel] = (),
    ) -> Callable[[Iterable[dict[str, Any]]], Iterable[Result | None]]:
        raise NotImplementedError
