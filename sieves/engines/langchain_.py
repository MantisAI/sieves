import asyncio
import enum
from collections.abc import Iterable
from typing import Any, TypeAlias

import langchain_core.language_models
import pydantic

from sieves.engines.core import Executable, PydanticEngine

Model: TypeAlias = langchain_core.language_models.BaseChatModel
PromptSignature: TypeAlias = pydantic.BaseModel
Result: TypeAlias = pydantic.BaseModel


class InferenceMode(enum.Enum):
    structured_output = "structured_output"


class LangChain(PydanticEngine[PromptSignature, Result, Model, InferenceMode]):
    """Engine for LangChain."""

    @property
    def inference_modes(self) -> type[InferenceMode]:
        return InferenceMode

    def build_executable(
        self,
        inference_mode: InferenceMode,
        prompt_template: str | None,  # noqa: UP007
        prompt_signature: type[PromptSignature] | PromptSignature,
        fewshot_examples: Iterable[pydantic.BaseModel] = tuple(),
    ) -> Executable[Result | None]:
        assert isinstance(prompt_signature, type)
        cls_name = self.__class__.__name__
        template = self._create_template(prompt_template)

        def execute(values: Iterable[dict[str, Any]]) -> Iterable[Result | None]:
            """Execute prompts with engine for given values.
            :param values: Values to inject into prompts.
            :return Iterable[Result | None]: Results for prompts. Results are None if corresponding prompt failed.
            """
            match inference_mode:
                case InferenceMode.structured_output:
                    model = self._model.with_structured_output(prompt_signature)

                    def generate(prompts: list[str]) -> Iterable[Result]:
                        try:
                            yield from asyncio.run(model.abatch(prompts, **self._inference_kwargs))
                        except pydantic.ValidationError as ex:
                            raise pydantic.ValidationError(
                                f"Encountered problem in parsing {cls_name} output. Double-check your prompts and "
                                f"examples."
                            ) from ex

                    generator = generate
                case _:
                    raise ValueError(f"Inference mode {inference_mode} not supported by {cls_name} engine.")

            yield from self._infer(generator, template, values, fewshot_examples)

        return execute
