"""LangChain model wrapper for structured outputs using Pydantic."""

import asyncio
import enum
from collections.abc import Iterable, Sequence
from typing import Any, override

import langchain_core.language_models
import nest_asyncio
import pydantic

from sieves.model_wrappers.core import Executable, PydanticModelWrapper
from sieves.model_wrappers.types import TokenUsage

nest_asyncio.apply()

Model = langchain_core.language_models.BaseChatModel
PromptSignature = pydantic.BaseModel
Result = pydantic.BaseModel


class InferenceMode(enum.Enum):
    """Available inference modes."""

    structured = "structured"


class LangChain(PydanticModelWrapper[PromptSignature, Result, Model, InferenceMode]):
    """ModelWrapper for LangChain."""

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
        fewshot_examples: Sequence[pydantic.BaseModel] = tuple(),
    ) -> Executable[Result | None]:
        assert isinstance(prompt_signature, type)
        cls_name = self.__class__.__name__
        template = self._create_template(prompt_template)
        model = self._model.with_structured_output(prompt_signature, include_raw=True)

        def execute(values: Sequence[dict[str, Any]]) -> Sequence[tuple[Result | None, Any, TokenUsage]]:
            """Execute prompts with model wrapper for given values.

            :param values: Values to inject into prompts.
            :return: Sequence of tuples containing results, raw outputs, and token usage. Results are None if
                corresponding prompt failed.
            """
            match inference_mode:
                case InferenceMode.structured:

                    def generate(prompts: list[str]) -> Iterable[tuple[Result, Any, TokenUsage]]:
                        try:
                            results = asyncio.run(model.abatch(prompts, **self._inference_kwargs))
                            for res in results:
                                usage = TokenUsage()
                                raw = res["raw"]
                                if hasattr(raw, "usage_metadata") and raw.usage_metadata:
                                    usage.input_tokens = raw.usage_metadata.get("input_tokens")
                                    usage.output_tokens = raw.usage_metadata.get("output_tokens")

                                yield res["parsed"], raw, usage

                        except Exception as err:
                            raise RuntimeError(
                                f"Encountered problem in parsing {cls_name} output. Double-check your prompts and "
                                f"examples."
                            ) from err

                    generator = generate
                case _:
                    raise ValueError(f"Inference mode {inference_mode} not supported by {cls_name} model wrapper.")

            return self._infer(generator, template, values, fewshot_examples)

        return execute
