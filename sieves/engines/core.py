import abc
import enum
from collections.abc import Callable, Iterable
from typing import Any, Generic, Protocol, TypeVar

PromptSignature = TypeVar("PromptSignature")
Model = TypeVar("Model")
Result = TypeVar("Result", covariant=True)
InferenceMode = TypeVar("InferenceMode", bound=enum.Enum)


class Executable(Protocol[Result]):
    def __call__(self, values: Iterable[dict[str, Any]]) -> Iterable[Result]:
        ...


class Engine(Generic[PromptSignature, Result, Model, InferenceMode]):
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
    def inference_modes(self) -> type[InferenceMode]:
        """Which inference modes are supported.
        :returns: Supported inference modes.
        """

    @abc.abstractmethod
    def build_executable(
        self,
        inference_mode: InferenceMode,
        prompt_template: str | None,
        prompt_signature: PromptSignature,
    ) -> Callable[[Iterable[dict[str, Any]]], Iterable[Result]]:
        """
        Returns prompt executable, i.e. a function that wraps an engine-native prediction generators. Such engine-native
        generators are e.g. Predict in DSPy, generator in outlines, Jsonformer in jsonformers).
        :param inference_mode: Inference mode to use (e.g. classification, JSON, ... - this is engine-specific).
        :param prompt_template: Prompt template.
        :param prompt_signature: Expected prompt signature.
        :return: Prompt executable.
        """
