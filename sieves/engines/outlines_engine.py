import enum
from typing import Any, Callable, Iterable, Optional, Type, TypeAlias

import outlines
import pydantic
from outlines.models import MLXLM, ExLlamaV2Model, LlamaCpp, OpenAI, Transformers, TransformersVision

from sieves.engines.core import Engine, Executable

PromptSignature: TypeAlias = pydantic.BaseModel | list[str] | str
Model: TypeAlias = ExLlamaV2Model | LlamaCpp | MLXLM | OpenAI | TransformersVision | Transformers
Result: TypeAlias = pydantic.BaseModel | str


class InferenceMode(enum.Enum):
    # For normal text output, i.e. no structured generation.
    text = (outlines.generate.text,)
    # For limited set of choices, e.g. classification.
    choice = (outlines.generate.choice,)
    # Regex-conforming output.
    regex = (outlines.generate.regex,)
    # Output conforming to Pydantic models.
    json = (outlines.generate.json,)


class Outlines(Engine[PromptSignature, Result, Model, InferenceMode]):
    @property
    def inference_modes(self) -> Type[InferenceMode]:
        return InferenceMode

    def build_executable(
        self,
        inference_mode: InferenceMode,
        prompt_template: Optional[str],
        prompt_signature: PromptSignature,
    ) -> Executable[Result]:
        assert prompt_template, ValueError("prompt_template has to be provided to Outlines engine by task.")

        def execute(values: Iterable[dict[str, Any]]) -> Iterable[Result]:
            generator_factory: Callable[..., Any] = inference_mode.value[0]

            match inference_mode:
                case InferenceMode.text:
                    generator = generator_factory(self._model)
                case InferenceMode.regex:
                    # PromptSignature is used as regex.
                    assert isinstance(prompt_signature, str), ValueError(
                        "PromptSignature has to be supplied as string in outlines regex mode."
                    )
                    generator = generator_factory(self._model, regex_str=prompt_signature)
                case InferenceMode.choice:
                    assert isinstance(prompt_signature, list), ValueError(
                        "PromptSignature has to be supplied as list of strings or enum values in outlines choice mode."
                    )
                    generator = generator_factory(self._model, choices=prompt_signature)
                case InferenceMode.json:
                    assert isinstance(prompt_signature, pydantic.BaseModel), ValueError(
                        "PromptSignature has to be supplied as Pydantic model in outlines json mode."
                    )
                    generator = generator_factory(self._model, schema_object=prompt_signature)
                case _:
                    raise ValueError(f"Inference mode {inference_mode} not supported by outlines engine.")

            return (generator(outlines.prompts.render(prompt_template, **doc_values)) for doc_values in values)

        return execute
