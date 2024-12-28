from typing import Any, Callable, Iterable, Optional, TypeAlias

import outlines
import pydantic
from outlines.models import MLXLM, ExLlamaV2Model, LlamaCpp, OpenAI, Transformers, TransformersVision

from sieves.engines import Engine

PromptSignature: TypeAlias = pydantic.BaseModel | list[str]
Model: TypeAlias = ExLlamaV2Model | LlamaCpp | MLXLM | OpenAI | TransformersVision | Transformers
InferenceGenerator: TypeAlias = outlines.generate.api.SequenceGeneratorAdapter
Result: TypeAlias = pydantic.BaseModel | str
Executable: TypeAlias = Callable[[Iterable[dict[str, Any]]], Iterable[Result]]


class Outlines(Engine[PromptSignature, Result, Model, InferenceGenerator]):
    def build_executable(
        self,
        inference_generator_factory: Callable[..., InferenceGenerator],
        prompt_template: str,
        prompt_signature: Optional[PromptSignature] = None,
    ) -> Executable:
        if prompt_signature:
            generator = inference_generator_factory(self.model, prompt_signature)
        else:
            generator = inference_generator_factory(self.model)

        def execute(values: Iterable[dict[str, Any]]) -> Iterable[Result]:
            return (generator(outlines.prompts.render(prompt_template, **doc_values)) for doc_values in values)

        return execute
