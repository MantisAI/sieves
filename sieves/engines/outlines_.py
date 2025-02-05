import enum
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias

import outlines
import pydantic
from outlines.models import MLXLM, ExLlamaV2Model, LlamaCpp, OpenAI, Transformers, TransformersVision

from sieves.engines.core import Executable, PydanticEngine

PromptSignature: TypeAlias = pydantic.BaseModel | list[str] | str
Model: TypeAlias = ExLlamaV2Model | LlamaCpp | MLXLM | OpenAI | TransformersVision | Transformers
Result: TypeAlias = pydantic.BaseModel | str


class InferenceMode(enum.Enum):
    """Available inference modes.
    Note: generator functions are wrapped in tuples, as otherwise the Enum instance seems to be replaced by the function
    itself - not sure why that happens. Should take another look at this.
    """

    # For normal text output, i.e. no structured generation.
    text = (outlines.generate.text,)
    # For limited set of choices, e.g. classification.
    choice = (outlines.generate.choice,)
    # Regex-conforming output.
    regex = (outlines.generate.regex,)
    # Output conforming to Pydantic models.
    json = (outlines.generate.json,)


class Outlines(PydanticEngine[PromptSignature, Result, Model, InferenceMode]):
    @property
    def inference_modes(self) -> type[InferenceMode]:
        return InferenceMode

    def build_executable(
        self,
        inference_mode: InferenceMode,
        prompt_template: str | None,  # noqa: UP007
        prompt_signature: type[PromptSignature] | PromptSignature,
        fewshot_examples: Iterable[pydantic.BaseModel] = (),
    ) -> Executable[Result | None]:
        cls_name = self.__class__.__name__
        template = self._create_template(prompt_template)

        def execute(values: Iterable[dict[str, Any]]) -> Iterable[Result | None]:
            """Execute prompts with engine for given values.
            :param values: Values to inject into prompts.
            :return Iterable[Result | None]: Results for prompts. Results are None if corresponding prompt failed.
            """
            generator_factory: Callable[..., Any] = inference_mode.value[0]

            match inference_mode:
                case InferenceMode.text:
                    seq_generator = generator_factory(self._model, **self._init_kwargs)
                case InferenceMode.regex:
                    assert isinstance(prompt_signature, str), ValueError(
                        "PromptSignature has to be supplied as string in outlines regex mode."
                    )
                    seq_generator = generator_factory(self._model, regex_str=prompt_signature, **self._init_kwargs)
                case InferenceMode.choice:
                    assert isinstance(prompt_signature, list), ValueError(
                        f"PromptSignature has to be supplied as list of strings or enum values in {cls_name} choice "
                        f"mode."
                    )
                    seq_generator = generator_factory(self._model, choices=prompt_signature, **self._init_kwargs)

                case InferenceMode.json:
                    assert isinstance(prompt_signature, type) and issubclass(prompt_signature, pydantic.BaseModel)
                    seq_generator = generator_factory(self._model, schema_object=prompt_signature, **self._init_kwargs)
                case _:
                    raise ValueError(f"Inference mode {inference_mode} not supported by {cls_name} engine.")

            def generate(prompts: list[str]) -> Iterable[Result]:
                yield from seq_generator(prompts, **self._inference_kwargs)

            yield from self._infer(
                generate,
                template,
                values,
                fewshot_examples,
            )

        return execute
