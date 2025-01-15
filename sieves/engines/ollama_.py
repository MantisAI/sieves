import enum
from collections.abc import Iterable
from typing import Any, TypeAlias

import jinja2
import ollama
import pydantic

from sieves.engines.core import Engine, Executable


class Model(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    name: str
    client: ollama.Client


PromptSignature: TypeAlias = type[pydantic.BaseModel]
Result: TypeAlias = pydantic.BaseModel


class InferenceMode(enum.Enum):
    chat = "chat"


class Ollama(Engine[PromptSignature, Result, Model, InferenceMode]):
    """Engine for Ollama.
    Make sure a Ollama server is running.
            In a nutshell:
            > curl -fsSL https://ollama.ai/install.sh | sh
            > ollama serve (or ollama run MODEL_ID)
    """

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
        fewshot_examples: Iterable[pydantic.BaseModel] = tuple(),
    ) -> Executable[Result]:
        cls_name = self.__class__.__name__
        assert prompt_signature, f"prompt_signature has to be provided to {cls_name}."
        template = jinja2.Template(prompt_template)

        def execute(values: Iterable[dict[str, Any]]) -> Iterable[Result]:
            match inference_mode:
                case InferenceMode.chat:

                    def generate(prompt: str, **inference_kwargs: dict[str, Any]) -> Result:
                        result = self._model.client.chat(
                            messages=[{"role": "user", "content": prompt}],
                            model=self._model.name,
                            format=prompt_signature.model_json_schema(),
                            **inference_kwargs,
                        )
                        try:
                            return prompt_signature.model_validate_json(result.message.content)
                        except pydantic.ValidationError as ex:
                            raise pydantic.ValidationError(
                                "Encountered problem in parsing Ollama output. Double-check your prompts and examples."
                            ) from ex

                    generator = generate
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
