import abc
from typing import Any, Generic, Iterable, Protocol, TypeVar

PromptTemplate = TypeVar("PromptTemplate")
PromptSignature = TypeVar("PromptSignature")
ExecutableResult = TypeVar("ExecutableResult", covariant=True)


class IExecutable(Protocol[ExecutableResult]):
    def __call__(self, values: Iterable[dict[str, Any]]) -> Iterable[ExecutableResult]:
        """
        Execute prompts.
        :param values: Sets of values to inject into prompt_template (one for each prompt execution).
        :return: Task results.
        """


Executable = TypeVar("Executable", bound=IExecutable[Any])


class Engine(Generic[PromptTemplate, PromptSignature, Executable]):
    def __init__(self, model_id: str, model_kwargs: dict[str, Any]):
        """
        :param model_id: ID of model to use.
        :param model_kwargs: Model init arguments.
        """
        self._model_id = model_id
        self._model_kwargs = model_kwargs

    @classmethod
    @abc.abstractmethod
    def convert_prompt_template(cls, prompt_template: str, variable_names: tuple[str] = ()) -> PromptTemplate:  # type: ignore[assignment]
        """Returns string prompt template in engine-native format.
        :param prompt_template: Template to convert.
        :param variable_names: Names of variables that will be injected into string. Note: not used for all engines -
        e.g. DSPy - but by others, e.g. outlines.
        :returns: Converted prompt template.
        """

    @abc.abstractmethod
    def build_executable(self, prompt_template: PromptTemplate, prompt_signature: PromptSignature) -> Executable:
        """
        Returns prompt executor, i.e. Predict in DSPy, Function in outlines, Jsonformer in jsonformers).
        :param prompt_template: Prompt template.
        :param prompt_signature: Expected prompt signature.
        :return: Prompt executable.
        """
