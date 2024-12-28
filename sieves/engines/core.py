import abc
from typing import Any, Callable, Generic, Iterable, Optional, TypeVar

PromptSignature = TypeVar("PromptSignature")
Model = TypeVar("Model")
Result = TypeVar("Result", covariant=True)
InferenceGenerator = TypeVar("InferenceGenerator")


class Engine(Generic[PromptSignature, Result, Model, InferenceGenerator]):
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

    @abc.abstractmethod
    def build_executable(
        self,
        inference_generatory_factory: Callable[..., InferenceGenerator],
        prompt_template: str,
        prompt_signature: Optional[PromptSignature] = None,
    ) -> Callable[[Iterable[dict[str, Any]]], Iterable[Result]]:
        """
        Returns prompt executable, i.e. a function that wraps an engine-native prediction generators. Such engine-native
        generators are e.g. Predict in DSPy, generator in outlines, Jsonformer in jsonformers).
        :param inference_generatory_factory: Callable returning engine-native inference generator to use.
        :param prompt_template: Prompt template.
        :param prompt_signature: Expected prompt signature.
        :return: Prompt executable.
        """
