"""Engine core interfaces and base classes used by backends."""

from __future__ import annotations

import abc
import asyncio
import enum
import itertools
import sys
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from typing import Any, Generic, Protocol, TypeVar, override

import instructor.exceptions
import jinja2
import pydantic

from sieves.engines.types import GenerationSettings

EnginePromptSignature = TypeVar("EnginePromptSignature")
EngineModel = TypeVar("EngineModel")
EngineResult = TypeVar("EngineResult", covariant=True)
EngineInferenceMode = TypeVar("EngineInferenceMode", bound=enum.Enum)


class Executable(Protocol[EngineResult]):
    """Callable protocol representing a compiled prompt executable."""

    def __call__(self, values: Iterable[dict[str, Any]]) -> Iterable[EngineResult | None]:
        """Execute prompt executable for given values.

        :param values: Values to inject into prompts.
        :return: Results for prompts.
        """
        ...


class Engine(Generic[EnginePromptSignature, EngineResult, EngineModel, EngineInferenceMode]):
    """Base class for engines wrapping model invocation and batching."""

    _MAX_TOKENS = 2**12

    def __init__(self, model: EngineModel, generation_settings: GenerationSettings):
        """Initialize engine with model and generation settings.

        :param model: Instantiated model instance.
        :param generation_settings: Generation settings.
        """
        self._model = model
        self._generation_settings = generation_settings
        self._inference_kwargs = generation_settings.inference_kwargs or {}
        self._init_kwargs = generation_settings.init_kwargs or {}
        self._strict_mode = generation_settings.strict_mode
        self._batch_size = generation_settings.batch_size

    @property
    def generation_settings(self) -> GenerationSettings:
        """Return generation settings.

        :return: Generation settings.
        """
        return self._generation_settings

    @property
    def model(self) -> EngineModel:
        """Return model instance.

        :return: Model instance.
        """
        return self._model

    @property
    @abc.abstractmethod
    def supports_few_shotting(self) -> bool:
        """Return whether engine supports few-shotting.

        :return: Whether engine supports few-shotting.
        """

    @property
    @abc.abstractmethod
    def inference_modes(self) -> type[EngineInferenceMode]:
        """Return supported inference modes.

        :return: Supported inference modes.
        """

    @abc.abstractmethod
    def build_executable(
        self,
        inference_mode: EngineInferenceMode,
        prompt_template: str | None,
        prompt_signature: type[EnginePromptSignature] | EnginePromptSignature,
        fewshot_examples: Iterable[pydantic.BaseModel] = (),
    ) -> Executable[EngineResult | None]:
        """Return a prompt executable for the given signature and mode.

        This wraps the engine‑native generation callable (e.g., DSPy Predict,
        Outlines Generator) with Sieves’ uniform interface.
        :param inference_mode: Inference mode to use (e.g. classification, JSON, ... - this is engine-specific).
        :param prompt_template: Prompt template.
        :param prompt_signature: Expected prompt signature type.
        :param fewshot_examples: Few-shot examples.
        :return: Prompt executable.
        """

    @staticmethod
    def _convert_fewshot_examples(fewshot_examples: Iterable[pydantic.BaseModel]) -> list[dict[str, Any]]:
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


class PydanticEngine(abc.ABC, Engine[EnginePromptSignature, EngineResult, EngineModel, EngineInferenceMode]):
    """Abstract super class for engines using Pydantic signatures and results.

    Note that this class also assumes the engine accepts a prompt. This holds true for most engines - it doesn't only
    for those with an idiocratic way to process prompts like DSPy, or decoder-only models which don't work with
    object-based signatures anyway.
    If and once we add support for a Pydantic-based engine that doesn't accept prompt templates, we'll adjust by
    modifying `_infer()` to accept an additional parameter specifying how to handle prompt/instruction injection (and
    we might have to make `supports_few_shotting()` engine-specific again).
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
        generator: Callable[[list[str]], Iterable[EngineResult]],
        template: jinja2.Template,
        values: Iterable[dict[str, Any]],
        fewshot_examples: Iterable[pydantic.BaseModel],
    ) -> Iterable[EngineResult | None]:
        """Run inference in batches with exception handling.

        :param generator: Callable generating responses.
        :param template: Prompt template.
        :param values: Doc values to inject.
        :param fewshot_examples: Fewshot examples.
        :return: Results parsed from responses.
        """
        fewshot_examples_dict = Engine._convert_fewshot_examples(fewshot_examples)
        examples = {"examples": fewshot_examples_dict} if len(fewshot_examples_dict) else {}
        batch_size = self._batch_size if self._batch_size != -1 else sys.maxsize
        # Ensure values are read as generator for standardized batch handling (otherwise we'd have to use different
        # batch handling depending on whether lists/tuples or generators are used).
        values = (v for v in values)

        while batch := [vals for vals in itertools.islice(values, batch_size)]:
            if len(batch) == 0:
                break

            try:
                yield from generator([template.render(**doc_values, **examples) for doc_values in batch])

            except (
                TypeError,
                pydantic.ValidationError,
                instructor.exceptions.InstructorRetryException,
                instructor.exceptions.IncompleteOutputException,
            ) as err:
                if self._strict_mode:
                    raise ValueError(
                        "Encountered problem when executing prompt. Ensure your few-shot examples and document "
                        "chunks contain sensible information."
                    ) from err
                else:
                    yield from (None for _ in range(len(batch)))
