import enum
from collections.abc import Iterable
from typing import Any, TypeAlias

import jinja2
import transformers

from sieves.engines.core import Engine, Executable

PromptSignature: TypeAlias = list[str]
Model: TypeAlias = transformers.Pipeline
Result: TypeAlias = dict[str, list[str] | list[float]]


class InferenceMode(enum.Enum):
    """Available inference modes."""

    default = 0


class HuggingFace(Engine[PromptSignature, Result, Model, InferenceMode]):
    @property
    def inference_modes(self) -> type[InferenceMode]:
        return InferenceMode

    @property
    def supports_few_shotting(self) -> bool:
        return False

    def build_executable(
        self,
        inference_mode: InferenceMode,
        prompt_template: str | None,
        prompt_signature: PromptSignature,
    ) -> Executable[Result]:
        cls_name = self.__class__.__name__
        assert prompt_template, ValueError(f"prompt_template has to be provided to {cls_name} engine by task.")

        def execute(values: Iterable[dict[str, Any]]) -> Iterable[Result]:
            match inference_mode:
                case InferenceMode.default:
                    template = jinja2.Template(prompt_template)

                    def generate(dv: dict[str, Any]) -> Result:
                        text = dv.pop("text")
                        result = self._model(
                            text,
                            prompt_signature,
                            # Render hypothesis template with everything but text.
                            hypothesis_template=template.render(**dv),
                            **self._inference_kwargs,
                        )
                        assert isinstance(result, dict)
                        return result

                    generator = generate
                case _:
                    raise ValueError(f"Inference mode {inference_mode} not supported by {cls_name} engine.")

            return (generator(doc_values) for doc_values in values)

        return execute
