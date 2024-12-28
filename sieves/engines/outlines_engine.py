import enum
from typing import Any, Callable, Iterable, Optional, Type, TypeAlias

import outlines
import pydantic
from outlines.models import MLXLM, ExLlamaV2Model, LlamaCpp, OpenAI, Transformers, TransformersVision

from sieves.engines.core import Engine

PromptSignature: TypeAlias = pydantic.BaseModel | list[str] | str
Model: TypeAlias = ExLlamaV2Model | LlamaCpp | MLXLM | OpenAI | TransformersVision | Transformers
Result: TypeAlias = pydantic.BaseModel | str
Executable: TypeAlias = Callable[[Iterable[dict[str, Any]]], Iterable[Result]]


class InferenceMode(enum.Enum):
    text = 0
    choice = 1
    regex = 2
    json = 3


class Outlines(Engine[PromptSignature, Result, Model, InferenceMode]):
    @property
    def inference_modes(self) -> Type[InferenceMode]:
        return InferenceMode

    def build_executable(
        self,
        inference_mode: InferenceMode,
        prompt_template: str,
        prompt_signature: Optional[PromptSignature] = None,
    ) -> Executable:
        def execute(values: Iterable[dict[str, Any]]) -> Iterable[Result]:
            match inference_mode:
                case InferenceMode.text:
                    # PromptSignature is ignored in text mode.
                    generator = outlines.generate.text(self._model)
                case InferenceMode.regex:
                    # PromptSignature is used as regex.
                    if not isinstance(prompt_signature, str):
                        raise ValueError("PromptSignature has to be supplied as string in outlines regex mode.")
                    generator = outlines.generate.regex(self._model, regex_str=prompt_signature)
                case InferenceMode.choice:
                    if not isinstance(prompt_signature, list):
                        raise ValueError(
                            "PromptSignature has to be supplied as list of strings or enum values in "
                            "outlines choice mode."
                        )
                    generator = outlines.generate.choice(self._model, choices=prompt_signature)
                case InferenceMode.json:
                    if not isinstance(prompt_signature, pydantic.BaseModel):
                        raise ValueError("PromptSignature has to be supplied as Pydantic model in outlines json mode.")
                    generator = outlines.generate.json(self._model, schema_object=prompt_signature)
                case _:
                    raise ValueError(f"Inference mode {inference_mode} not supported by outlines engine.")

            return (generator(outlines.prompts.render(prompt_template, **doc_values)) for doc_values in values)

        return execute
