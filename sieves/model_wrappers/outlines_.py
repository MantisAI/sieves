"""Outlines model_wrapper."""

import asyncio
import enum
from collections.abc import Iterable, Sequence
from typing import Any, Literal, override

import json_repair
import outlines
import pydantic
from outlines.models import AsyncBlackBoxModel, BlackBoxModel, SteerableModel

from sieves.model_wrappers.core import Executable, PydanticModelWrapper
from sieves.model_wrappers.types import TokenUsage

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


class Outlines(PydanticModelWrapper[PromptSignature, Result, Model, InferenceMode]):
    """ModelWrapper for Outlines with multiple structured inference modes."""

    @override
    @property
    def inference_modes(self) -> type[InferenceMode]:
        return InferenceMode

    async def _generate_async(
        self,
        generator: (
            outlines.generator.SteerableGenerator
            | outlines.generator.BlackBoxGenerator
            | outlines.generator.AsyncBlackBoxGenerator
        ),
        prompt: str,
    ) -> Result | None:
        """Generate result async.

        :param generator: Generator instance to use for generation.
        :param prompt: Prompt to generate result for.
        :return: Result for prompt. Results are None if corresponding prompt failed.
        """
        result = generator(prompt, **self._inference_kwargs)
        assert isinstance(result, Result) or result is None
        return result

    @override
    def build_executable(
        self,
        inference_mode: InferenceMode,
        prompt_template: str | None,  # noqa: UP007
        prompt_signature: type[PromptSignature] | PromptSignature,
        fewshot_examples: Sequence[pydantic.BaseModel] = (),
    ) -> Executable[Result | None]:
        template = self._create_template(prompt_template)

        # Create Generator instance responsible for generating non-parsed text.
        if isinstance(prompt_signature, list):
            prompt_signature = Literal[*prompt_signature]  # type: ignore[invalid-type-form]

        if inference_mode == InferenceMode.regex:
            prompt_signature = outlines.types.Regex(prompt_signature)

        generator = outlines.Generator(self._model, output_type=prompt_signature, **self._init_kwargs)

        def execute(values: Sequence[dict[str, Any]]) -> Sequence[tuple[Result | None, Any, TokenUsage]]:
            """Execute prompts with model wrapper for given values.

            :param values: Values to inject into prompts.
            :return: Sequence of tuples containing results, raw outputs, and token usage. Results are None if
                corresponding prompt failed.
            """

            def generate(prompts: list[str]) -> Iterable[tuple[Result, Any, TokenUsage]]:
                try:
                    results = generator.batch(prompts, **self._inference_kwargs)
                # Batch mode is not implemented for all Outlines wrappers. Fall back to single-prompt mode in
                # that case.
                except NotImplementedError:
                    calls = [self._generate_async(generator, prompt) for prompt in prompts]
                    results = asyncio.run(self._execute_async_calls(calls))

                # Estimate token usage if tokenizer is available.
                tokenizer = self._get_tokenizer()

                if inference_mode == InferenceMode.json:
                    assert len(results) == len(prompts)
                    assert isinstance(prompt_signature, type) and issubclass(prompt_signature, pydantic.BaseModel)

                    for prompt, result in zip(prompts, results):
                        usage = TokenUsage(
                            input_tokens=self._count_tokens(prompt, tokenizer),
                            output_tokens=self._count_tokens(result, tokenizer),
                        )

                        try:
                            parsed = prompt_signature.model_validate_json(result)
                            yield parsed, result, usage
                        # If naive parsing fails: JSON is potentially invalid. Attempt to repair it, then try again.
                        except pydantic.ValidationError:
                            repaired = json_repair.repair_json(result)
                            parsed = prompt_signature.model_validate_json(repaired)
                            yield parsed, result, usage

                else:
                    for prompt, result in zip(prompts, results):
                        usage = TokenUsage(
                            input_tokens=self._count_tokens(prompt, tokenizer),
                            output_tokens=self._count_tokens(str(result), tokenizer),
                        )

                        yield result, result, usage

            return self._infer(
                generate,
                template,
                values,
                fewshot_examples,
            )

        return execute

    @override
    def _get_tokenizer(self) -> Any | None:
        # Outlines models usually have a tokenizer, but some wrappers (like OpenAI) might hide it.
        tokenizer = getattr(self._model, "tokenizer", None)
        if not tokenizer and hasattr(self._model, "model"):
            tokenizer = getattr(self._model.model, "tokenizer", None)

        # Fall back to tiktoken for remote OpenAI-compatible models if no tokenizer found.
        if not tokenizer:
            try:
                import tiktoken

                # Default to o200k_base for newer models if specific model encoding not found.
                tokenizer = tiktoken.get_encoding("o200k_base")
            except ImportError:
                tokenizer = None

        return tokenizer
