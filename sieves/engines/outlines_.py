import enum
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias

import jinja2
import outlines
import pydantic
from outlines.models import MLXLM, ExLlamaV2Model, LlamaCpp, OpenAI, Transformers, TransformersVision

from sieves.engines.core import Engine, Executable

PromptSignature: TypeAlias = type[pydantic.BaseModel] | list[str] | str
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


class Outlines(Engine[PromptSignature, Result, Model, InferenceMode]):
    @property
    def inference_modes(self) -> type[InferenceMode]:
        return InferenceMode

    @property
    def supports_few_shotting(self) -> bool:
        return True

    def build_executable(
        self,
        inference_mode: InferenceMode,
        prompt_template: str | None,  # noqa: UP007
        prompt_signature: PromptSignature,
        fewshot_examples: Iterable[pydantic.BaseModel] = (),
    ) -> Executable[Result]:
        cls_name = self.__class__.__name__
        assert prompt_template, ValueError(f"prompt_template has to be provided to {cls_name} engine by task.")
        template = jinja2.Template(prompt_template)

        def execute(values: Iterable[dict[str, Any]]) -> Iterable[Result]:
            generator_factory: Callable[..., Any] = inference_mode.value[0]

            match inference_mode:
                case InferenceMode.text:
                    generator = generator_factory(self._model, **self._init_kwargs)
                case InferenceMode.regex:
                    assert isinstance(prompt_signature, str), ValueError(
                        "PromptSignature has to be supplied as string in outlines regex mode."
                    )
                    generator = generator_factory(self._model, regex_str=prompt_signature, **self._init_kwargs)
                case InferenceMode.choice:
                    assert isinstance(prompt_signature, list), ValueError(
                        f"PromptSignature has to be supplied as list of strings or enum values in {cls_name} choice "
                        f"mode."
                    )
                    generator = generator_factory(self._model, choices=prompt_signature, **self._init_kwargs)
                case InferenceMode.json:
                    assert isinstance(prompt_signature, type) and issubclass(prompt_signature, pydantic.BaseModel)
                    generator = generator_factory(self._model, schema_object=prompt_signature, **self._init_kwargs)
                case _:
                    raise ValueError(f"Inference mode {inference_mode} not supported by {cls_name} engine.")

            # Convert few-shot examples.
            fs_examples: list[dict[str, Any]] = []
            for fs_example in fewshot_examples:
                fs_examples.append(fs_example.model_dump(serialize_as_any=True))

            return (
                generator(
                    template.render(**doc_values, **({"examples": fs_examples} if len(fs_examples) else {})),
                    **self._inference_kwargs,
                )
                for doc_values in values
            )

        return execute
