import abc
import enum
from collections.abc import Callable, Iterable
from typing import Any, Generic, Protocol, TypeVar

import pydantic

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
