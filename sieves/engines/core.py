import abc
import enum
from typing import Any, Callable, Generic, Iterable, Optional, Protocol, Type, TypeVar

PromptSignature = TypeVar("PromptSignature")
Model = TypeVar("Model")
Result = TypeVar("Result", covariant=True)
InferenceMode = TypeVar("InferenceMode", bound=enum.Enum)


class Executable(Protocol[Result]):
    def __call__(self, values: Iterable[dict[str, Any]]) -> Iterable[Result]:
        ...


class Engine(Generic[PromptSignature, Result, Model, InferenceMode]):
    def __init__(self, model: Model):
        """
        :param model: Instantiated model instance.
        """
        self._model = model

    @property
    def model(self) -> Model:
        """Return model instance.
        :returns: Model instance.
        """
        return self._model

    @property
    @abc.abstractmethod
    def inference_modes(self) -> Type[InferenceMode]:
        """Supported inference modes."""

    @abc.abstractmethod
    def build_executable(
        self,
        inference_mode: InferenceMode,
        prompt_template: Optional[str],
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
