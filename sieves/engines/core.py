from __future__ import annotations

import abc
import asyncio
import enum
import itertools
import sys
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from typing import Any, Generic, Protocol, TypeVar

import instructor.exceptions
import jinja2
import pydantic

from sieves.serialization import Attribute, Config

EnginePromptSignature = TypeVar("EnginePromptSignature")
EngineModel = TypeVar("EngineModel")
EngineResult = TypeVar("EngineResult", covariant=True)
EngineInferenceMode = TypeVar("EngineInferenceMode", bound=enum.Enum)


class Executable(Protocol[EngineResult]):
    def __call__(self, values: Iterable[dict[str, Any]]) -> Iterable[EngineResult | None]:
        ...


class InternalEngine(Generic[EnginePromptSignature, EngineResult, EngineModel, EngineInferenceMode]):
    _MAX_TOKENS = 2**12

    def __init__(
        self,
        model: EngineModel,
        init_kwargs: dict[str, Any] | None,
        inference_kwargs: dict[str, Any] | None,
        strict_mode: bool,
        batch_size: int,
    ):
        """
        :param model: Instantiated model instance.
        :param init_kwargs: Optional kwargs to supply to engine executable at init time.
        :param inference_kwargs: Optional kwargs to supply to engine executable at inference time.
        :param strict_mode: If True, exception is raised if prompt response can't be parsed correctly.
        :param batch_size: Batch size in processing prompts. -1 will batch all documents in one go. Not all engines
            support batching.
        """
        self._model = model
        self._inference_kwargs = inference_kwargs or {}
        self._init_kwargs = init_kwargs or {}
        self._strict_mode = strict_mode
        self._batch_size = batch_size

    @property
    def model(self) -> EngineModel:
        """Return model instance.
        :return: Model instance.
        """
        return self._model

    @property
    @abc.abstractmethod
    def supports_few_shotting(self) -> bool:
        """Whether engine supports few-shotting. If not, only zero-shotting is supported.
        :return: Whether engine supports few-shotting.
        """

    @property
    @abc.abstractmethod
    def inference_modes(self) -> type[EngineInferenceMode]:
        """Which inference modes are supported.
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
        """
        Returns prompt executable, i.e. a function that wraps an engine-native prediction generators. Such engine-native
        generators are e.g. Predict in DSPy, generator in outlines, Jsonformer in jsonformers).
        :param inference_mode: Inference mode to use (e.g. classification, JSON, ... - this is engine-specific).
        :param prompt_template: Prompt template.
        :param prompt_signature: Expected prompt signature type.
        :param fewshot_examples: Few-shot examples.
        :return: Prompt executable.
        """

    @staticmethod
    def _convert_fewshot_examples(fewshot_examples: Iterable[pydantic.BaseModel]) -> list[dict[str, Any]]:
        """
        Convert fewshot examples from pydantic.BaseModel instance to dicts.
        :param fewshot_examples: Fewshot examples to convert.
        :return: Fewshot examples as dicts.
        """
        return [fs_example.model_dump(serialize_as_any=True) for fs_example in fewshot_examples]

    @property
    def _attributes(self) -> dict[str, Attribute]:
        """Returns attributes to serialize.
        :return: Dict of attributes to serialize.
        """
        # Note: init_kwargs and inference_kwargs are potentially unfit for serialization as they contain arbitrary
        # objects.
        return {
            "model": Attribute(value=self._model),
            "init_kwargs": Attribute(value=self._init_kwargs),
            "inference_kwargs": Attribute(value=self._inference_kwargs),
            "strict_mode": Attribute(value=self._strict_mode),
            "batch_size": Attribute(value=self._batch_size),
        }

    def serialize(self) -> Config:
        """Serializes engine.
        :return: Config instance.
        """
        return Config.create(self.__class__, self._attributes)

    @classmethod
    def deserialize(
        cls, config: Config, **kwargs: dict[str, Any]
    ) -> InternalEngine[EnginePromptSignature, EngineResult, EngineModel, EngineInferenceMode]:
        """Generate Engine instance from config.
        :param config: Config to generate instance from.
        :param kwargs: Values to inject into loaded config.
        :return: Deserialized Engine instance.
        """
        return cls(**config.to_init_dict(cls, **kwargs))

    @staticmethod
    async def _execute_async_calls(calls: list[Coroutine[Any, Any, Any]] | list[Awaitable[Any]]) -> Any:
        """Executes batch of async functions.
        :param calls: Async calls to execute.
        :return: Parsed response objects.
        """
        return await asyncio.gather(*calls)


class PydanticEngine(abc.ABC, InternalEngine[EnginePromptSignature, EngineResult, EngineModel, EngineInferenceMode]):
    """Abstract super class for all engines working directly with Pydantic objects for prompt signatures and results.
    Note that this class also assumes the engine accepts a prompt. This holds true for most engines - it doesn't only
    for those with an idiocratic way to process prompts like DSPy, or decoder-only models which don't work with
    object-based signatures anyway.
    If and once we add support for a Pydantic-based engine that doesn't accept prompt templates, we'll adjust by
    modifying `_infer()` to accept an additional parameter specifying how to handle prompt/instruction injection (and
    we might have to make `supports_few_shotting()` engine-specific again).
    """

    @classmethod
    def _create_template(cls, template: str | None) -> jinja2.Template:
        """Creates Jinja2 template from template string.
        :param template: Template string.
        :return: Jinja2 template.
        """
        assert template, f"prompt_template has to be provided to {cls.__name__}."
        return jinja2.Template(template)

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
        """
        Runs inference record by record with exception handling for template- and Pydantic-based engines.
        :param generator: Callable generating responses.
        :param template: Prompt template.
        :param values: Doc values to inject.
        :param fewshot_examples: Fewshot examples.
        :return: Results parsed from responses.
        """
        fewshot_examples_dict = InternalEngine._convert_fewshot_examples(fewshot_examples)
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
