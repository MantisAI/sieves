"""ModelWrapper core interfaces and base classes used by backends."""

from __future__ import annotations

import abc
import asyncio
import enum
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Sequence
from typing import Any, Protocol, TypeVar, override

import jinja2
import pydantic

from sieves.model_wrappers.types import ModelSettings, TokenUsage

ModelWrapperPromptSignature = TypeVar("ModelWrapperPromptSignature")
ModelWrapperModel = TypeVar("ModelWrapperModel")
ModelWrapperResult = TypeVar("ModelWrapperResult", covariant=True)
ModelWrapperInferenceMode = TypeVar("ModelWrapperInferenceMode", bound=enum.Enum)


class Executable(Protocol[ModelWrapperResult]):
    """Callable protocol representing a compiled prompt executable."""

    def __call__(self, values: Sequence[dict[str, Any]]) -> Sequence[tuple[ModelWrapperResult | None, Any, TokenUsage]]:
        """Execute prompt executable for given values.

        :param values: Values to inject into prompts.
        :return: Sequence of tuples containing (result, raw_output, usage) for prompts.
        """
        ...


class ModelWrapper[ModelWrapperPromptSignature, ModelWrapperResult, ModelWrapperModel, ModelWrapperInferenceMode]:
    """Base class for model wrappers handling model invocation and structured generation."""

    def __init__(self, model: ModelWrapperModel, model_settings: ModelSettings):
        """Initialize model wrapper with model and model settings.

        :param model: Instantiated model instance.
        :param model_settings: Model settings.
        """
        self._model = model
        self._model_settings = model_settings
        self._inference_kwargs = model_settings.inference_kwargs or {}
        self._init_kwargs = model_settings.init_kwargs or {}
        self._strict = model_settings.strict

    @property
    def model_settings(self) -> ModelSettings:
        """Return model settings.

        :return: Model settings.
        """
        return self._model_settings

    @property
    def model(self) -> ModelWrapperModel:
        """Return model instance.

        :return: Model instance.
        """
        return self._model

    @property
    @abc.abstractmethod
    def supports_few_shotting(self) -> bool:
        """Return whether model wrapper supports few-shotting.

        :return: Whether model wrapper supports few-shotting.
        """

    @property
    @abc.abstractmethod
    def inference_modes(self) -> type[ModelWrapperInferenceMode]:
        """Return supported inference modes.

        :return: Supported inference modes.
        """

    @abc.abstractmethod
    def build_executable(
        self,
        inference_mode: ModelWrapperInferenceMode,
        prompt_template: str | None,
        prompt_signature: type[ModelWrapperPromptSignature] | ModelWrapperPromptSignature,
        fewshot_examples: Sequence[pydantic.BaseModel] = (),
    ) -> Executable[ModelWrapperResult | None]:
        """Return a prompt executable for the given signature and mode.

        This wraps the model type‑native generation callable (e.g., DSPy Predict, Outlines Generator) with sieves’
        uniform interface.

        :param inference_mode: Inference mode to use (e.g. classification, JSON, ... - this is model type-specific).
        :param prompt_template: Prompt template.
        :param prompt_signature: Expected prompt signature type.
        :param fewshot_examples: Few-shot examples.
        :return: Prompt executable.
        """

    @staticmethod
    def convert_fewshot_examples(fewshot_examples: Sequence[pydantic.BaseModel]) -> list[dict[str, Any]]:
        """Convert few‑shot examples to dicts.

        :param fewshot_examples: Fewshot examples to convert.
        :return: Fewshot examples as dicts.
        """
        return [fs_example.model_dump(serialize_as_any=True) for fs_example in fewshot_examples]

    @staticmethod
    async def _execute_async_calls(calls: list[Coroutine[Any, Any, Any]] | list[Awaitable[Any]]) -> Any:
        """Execute a batch of async functions.

        :param calls: Async calls to execute.
        :return: Parsed response objects.
        """
        return await asyncio.gather(*calls)

    def _get_tokenizer(self) -> Any | None:
        """Return the tokenizer instance for this model if available.

        :return: Tokenizer instance or None.
        """
        return None

    def _count_tokens(self, text: str | None, tokenizer: Any | None = None) -> int | None:
        """Count tokens in a string using the provided or default tokenizer.

        :param text: Text to count tokens for.
        :param tokenizer: Optional tokenizer to use. If not provided, uses _get_tokenizer().
        :return: Token count or None if no tokenizer is available.
        """
        if text is None:
            return None

        tokenizer = tokenizer or self._get_tokenizer()
        if tokenizer:
            try:
                # Handle both standard transformers and tiktoken-style tokenizers.
                encoded = tokenizer.encode(text)
                return len(encoded)
            except Exception:
                return None
        return None


class PydanticModelWrapper(
    abc.ABC, ModelWrapper[ModelWrapperPromptSignature, ModelWrapperResult, ModelWrapperModel, ModelWrapperInferenceMode]
):
    """Abstract super class for model wrappers using Pydantic signatures and results.

    Note that this class also assumes the model wrapper accepts a prompt. This holds true for most model wrappers - it
    doesn't only for those with an idiosyncratic way to process prompts like DSPy, or decoder-only models which don't
    work with object-based signatures anyway.

    If and once we add support for a Pydantic-based model wrapper that doesn't accept prompt templates, we'll adjust by
    modifying `_infer()` to accept an additional parameter specifying how to handle prompt/instruction injection (and
    we might have to make `supports_few_shotting()` model type-specific again).
    """

    @classmethod
    def _create_template(cls, template: str | None) -> jinja2.Template:
        """Create Jinja2 template from template string.

        :param template: Template string.
        :return: Jinja2 template.
        """
        assert template, f"prompt_template has to be provided to {cls.__name__}."
        return jinja2.Template(template)

    @override
    @property
    def supports_few_shotting(self) -> bool:
        return True

    def _infer(
        self,
        generator: Callable[[list[str]], Iterable[tuple[ModelWrapperResult, Any, TokenUsage]]],
        template: jinja2.Template,
        values: Sequence[dict[str, Any]],
        fewshot_examples: Sequence[pydantic.BaseModel],
    ) -> Sequence[tuple[ModelWrapperResult | None, Any, TokenUsage]]:
        """Run inference in batches with exception handling.

        :param generator: Callable generating responses.
        :param template: Prompt template.
        :param values: Doc values to inject.
        :param fewshot_examples: Fewshot examples.
        :return: Sequence of tuples containing results parsed from responses, raw outputs, and token usage.
        """
        fewshot_examples_dict = ModelWrapper.convert_fewshot_examples(fewshot_examples)
        examples = {"examples": fewshot_examples_dict} if len(fewshot_examples_dict) else {}

        try:
            return list(generator([template.render(**doc_values, **examples) for doc_values in values]))

        except Exception as err:
            if self._strict:
                raise RuntimeError(
                    "Encountered problem when executing prompt. Ensure your few-shot examples and document "
                    "chunks contain sensible information."
                ) from err
            else:
                return [(None, None, TokenUsage()) for _ in range(len(values))]
