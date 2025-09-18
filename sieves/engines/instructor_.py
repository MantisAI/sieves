"""Instructor engine wrapper for structured outputs via response_model parsing."""

import asyncio
import enum
from collections.abc import Iterable
from typing import Any, TypeAlias, override

import instructor
import pydantic

from sieves.engines.core import Executable, PydanticEngine


class Model(pydantic.BaseModel):
    """Container for Instructor client configuration and model name."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    name: str
    client: instructor.AsyncInstructor


PromptSignature: TypeAlias = pydantic.BaseModel
Result: TypeAlias = pydantic.BaseModel


class InferenceMode(enum.Enum):
    """Available inference modes."""

    structured = "structured"


class Instructor(PydanticEngine[PromptSignature, Result, Model, InferenceMode]):
    """Engine for Instructor to obtain Pydantic‑validated structured outputs."""

    @override
    @property
    def inference_modes(self) -> type[InferenceMode]:
        return InferenceMode

    @override
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
                case InferenceMode.structured:

                    def generate(prompts: list[str]) -> Iterable[Result]:
                        calls = [
                            self._model.client.chat.completions.create(
                                messages=[{"role": "user", "content": prompt}],
                                model=self._model.name,
                                response_model=prompt_signature,
                                **({"max_tokens": Instructor._MAX_TOKENS} | self._inference_kwargs),
                            )
                            for prompt in prompts
                        ]
                        responses = asyncio.run(self._execute_async_calls(calls))

                        yield from responses

                case _:
                    raise ValueError(f"Inference mode {inference_mode} not supported by {cls_name} engine.")

            yield from self._infer(generate, template, values, fewshot_examples)

        return execute
