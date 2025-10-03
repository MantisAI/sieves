"""Outlines engine wrapper supporting text, choices, regex and JSON schemas."""

import enum
from collections.abc import Iterable
from typing import Any, Literal, override

import outlines
import pydantic
from outlines.models import AsyncBlackBoxModel, BlackBoxModel, SteerableModel

from sieves.engines.core import Executable, PydanticEngine

PromptSignature = (
    pydantic.BaseModel | list[str] | str | outlines.types.Choice | outlines.types.Regex | outlines.types.JsonSchema
)
Model = AsyncBlackBoxModel | BlackBoxModel | SteerableModel
Result = pydantic.BaseModel | str


class InferenceMode(enum.Enum):
    """Available inference modes.

    Note: generator functions are wrapped in tuples, as otherwise the Enum instance seems to be replaced by the function
    itself - not sure why that happens. Should take another look at this.
    """

    # For normal text output, i.e. no structured generation.
    text = "text"
    # For limited set of choices, e.g. classification.
    choice = "choice"
    # Regex-conforming output.
    regex = "regex"
    # Output conforming to Pydantic models.
    json = "json"


class Outlines(PydanticEngine[PromptSignature, Result, Model, InferenceMode]):
    """Engine for Outlines with multiple structured inference modes."""

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
        fewshot_examples: Iterable[pydantic.BaseModel] = (),
    ) -> Executable[Result | None]:
        template = self._create_template(prompt_template)

        # Create Generator instance responsible for generating non-parsed text.
        if isinstance(prompt_signature, list):
            prompt_signature = Literal[*prompt_signature]

        if inference_mode == InferenceMode.regex:
            prompt_signature = outlines.types.Regex(prompt_signature)

        generator = outlines.Generator(self._model, output_type=prompt_signature, **self._init_kwargs)

        def execute(values: Iterable[dict[str, Any]]) -> Iterable[Result | None]:
            """Execute prompts with engine for given values.

            :param values: Values to inject into prompts.
            :return Iterable[Result | None]: Results for prompts. Results are None if corresponding prompt failed.
            """

            def generate(prompts: list[str]) -> Iterable[Result]:
                try:
                    results = generator.batch(prompts, **self._inference_kwargs)
                # Batch mode is not implemented for all Outlines wrappers. Fall back to single-prompt mode in that case.
                except NotImplementedError:

                    async def generate_async(prompt: str) -> Result | None:
                        """Generate result async.

                        :param prompt: Prompt to generate result for.
                        :return: Result for prompt. Results are None if corresponding prompt failed.
                        """
                        return generator(prompt, **self._inference_kwargs)

                    calls = [generate_async(prompt) for prompt in prompts]
                    results = self._execute_async_calls(calls)

                if inference_mode == InferenceMode.json:
                    assert len(results) == len(prompts)
                    assert isinstance(prompt_signature, type) and issubclass(prompt_signature, pydantic.BaseModel)
                    yield from [prompt_signature.model_validate_json(result) for result in results]
                else:
                    yield from results

            yield from self._infer(
                generate,
                template,
                values,
                fewshot_examples,
            )

        return execute
