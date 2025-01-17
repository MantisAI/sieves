import enum
from collections.abc import Iterable
from typing import Any, TypeAlias

import jinja2
import langchain_core.language_models
import pydantic

from sieves.engines.core import Engine, Executable

Model: TypeAlias = langchain_core.language_models.BaseChatModel
PromptSignature: TypeAlias = type[pydantic.BaseModel]
Result: TypeAlias = pydantic.BaseModel


class InferenceMode(enum.Enum):
    structured_output = "structured_output"


class LangChain(Engine[PromptSignature, Result, Model, InferenceMode]):
    """Engine for LangChain."""

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
        assert prompt_template, f"prompt_template has to be provided to {cls_name}."
        template = jinja2.Template(prompt_template)

        def execute(values: Iterable[dict[str, Any]]) -> Iterable[Result]:
            match inference_mode:
                case InferenceMode.structured_output:
                    model = self._model.with_structured_output(prompt_signature)

                    def generate(prompt: str, **inference_kwargs: dict[str, Any]) -> Result:
                        try:
                            result = model.invoke(prompt, **inference_kwargs)
                            assert isinstance(result, Result)
                            return result
                        except pydantic.ValidationError as ex:
                            raise pydantic.ValidationError(
                                "Encountered problem in parsing Ollama output. Double-check your prompts and examples."
                            ) from ex

                    generator = generate
                case _:
                    raise ValueError(f"Inference mode {inference_mode} not supported by {cls_name} engine.")

            fewshot_examples_dict = LangChain._convert_fewshot_examples(fewshot_examples)
            return (
                generator(
                    template.render(**doc_values, **({"examples": fewshot_examples_dict})),
                    **self._inference_kwargs,
                )
                for doc_values in values
            )

        return execute
