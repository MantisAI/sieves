from __future__ import annotations

import abc
import enum
from collections.abc import Callable, Iterable
from typing import Any, Generic, Protocol, TypeVar

import pydantic

from sieves.serialization import Attribute, Config

EnginePromptSignature = TypeVar("EnginePromptSignature")
Model = TypeVar("Model")
EngineResult = TypeVar("EngineResult", covariant=True)
EngineInferenceMode = TypeVar("EngineInferenceMode", bound=enum.Enum)


class Executable(Protocol[EngineResult]):
    def __call__(self, values: Iterable[dict[str, Any]]) -> Iterable[EngineResult]:
        ...


class Engine(Generic[EnginePromptSignature, EngineResult, Model, EngineInferenceMode]):
    def __init__(
        self, model: Model, init_kwargs: dict[str, Any] | None = None, inference_kwargs: dict[str, Any] | None = None
    ):
        """
        :param model: Instantiated model instance.
        :param init_kwargs: Optional kwargs to supply to engine executable at init time.
        :param inference_kwargs: Optional kwargs to supply to engine executable at inference time.
        """
        self._model = model
        self._inference_kwargs = inference_kwargs or {}
        self._init_kwargs = init_kwargs or {}

    @property
    def model(self) -> Model:
        """Return model instance.
        :returns: Model instance.
        """
        return self._model

    @property
    @abc.abstractmethod
    def supports_few_shotting(self) -> bool:
        """Whether engine supports few-shotting. If not, only zero-shotting is supported.
        :returns: Whether engine supports few-shotting.
        """

    @property
    @abc.abstractmethod
    def inference_modes(self) -> type[EngineInferenceMode]:
        """Which inference modes are supported.
        :returns: Supported inference modes.
        """

    @abc.abstractmethod
    def build_executable(
        self,
        inference_mode: EngineInferenceMode,
        prompt_template: str | None,
        prompt_signature: EnginePromptSignature,
        fewshot_examples: Iterable[pydantic.BaseModel] = (),
    ) -> Callable[[Iterable[dict[str, Any]]], Iterable[EngineResult]]:
        """
        Returns prompt executable, i.e. a function that wraps an engine-native prediction generators. Such engine-native
        generators are e.g. Predict in DSPy, generator in outlines, Jsonformer in jsonformers).
        :param inference_mode: Inference mode to use (e.g. classification, JSON, ... - this is engine-specific).
        :param prompt_template: Prompt template.
        :param prompt_signature: Expected prompt signature.
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
        :returns: Dict of attributes to serialize.
        """
        # Note: init_kwargs and inference_kwargs are potentially unfit for serialization as they contain arbitrary
        # objects.
        return {
            "model": Attribute(value=self._model, is_placeholder=True),
            "init_kwargs": Attribute(value=self._init_kwargs, is_placeholder=False),
            "inference_kwargs": Attribute(value=self._inference_kwargs, is_placeholder=False),
        }

    def serialize(self) -> Config:
        """Serializes engine.
        :returns: Config instance.
        """
        return Config.create(self.__class__, self._attributes)

    @classmethod
    def deserialize(
        cls, config: Config, **kwargs: dict[str, Any]
    ) -> Engine[EnginePromptSignature, EngineResult, Model, EngineInferenceMode]:
        """Generate Engine instance from config.
        :param config: Config to generate instance from.
        :param kwargs: Values to inject into loaded config.
        :returns: Deserialized Engine instance.
        """
        return cls(**config.to_init_dict(cls, **kwargs))
