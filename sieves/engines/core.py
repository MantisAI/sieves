from __future__ import annotations

import abc
import enum
from collections.abc import Callable, Iterable
from typing import Any, Generic, Protocol, TypeVar

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


class Engine(Generic[EnginePromptSignature, EngineResult, EngineModel, EngineInferenceMode]):
    def __init__(
        self,
        model: EngineModel,
        init_kwargs: dict[str, Any] | None = None,
        inference_kwargs: dict[str, Any] | None = None,
        strict_mode: bool = False,
    ):
        """
        :param model: Instantiated model instance.
        :param init_kwargs: Optional kwargs to supply to engine executable at init time.
        :param inference_kwargs: Optional kwargs to supply to engine executable at inference time.
        :param strict_mode: If True, exception is raised if prompt response can't be parsed correctly.
        """
        self._model = model
        self._inference_kwargs = inference_kwargs or {}
        self._init_kwargs = init_kwargs or {}
        self._strict_mode = strict_mode

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
        }

    def serialize(self) -> Config:
        """Serializes engine.
        :return: Config instance.
        """
        return Config.create(self.__class__, self._attributes)

    @classmethod
    def deserialize(
        cls, config: Config, **kwargs: dict[str, Any]
    ) -> Engine[EnginePromptSignature, EngineResult, EngineModel, EngineInferenceMode]:
        """Generate Engine instance from config.
        :param config: Config to generate instance from.
        :param kwargs: Values to inject into loaded config.
        :return: Deserialized Engine instance.
        """
        return cls(**config.to_init_dict(cls, **kwargs))


class TemplateBasedEngine(abc.ABC, Engine[EnginePromptSignature, EngineResult, EngineModel, EngineInferenceMode]):
    @classmethod
    def _create_template(cls, template: str | None) -> jinja2.Template:
        """Creates Jinja2 template from template string.
        :param template: Template string.
        :return: Jinja2 template.
        """
        assert template, f"prompt_template has to be provided to {cls.__name__}."
        return jinja2.Template(template)

    def _infer(
        self,
        generator: Callable[..., EngineResult],
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
        fewshot_examples_dict = Engine._convert_fewshot_examples(fewshot_examples)

        for doc_values in values:
            try:
                yield generator(
                    template.render(
                        **doc_values, **({"examples": fewshot_examples_dict} if len(fewshot_examples_dict) else {})
                    ),
                    **self._inference_kwargs,
                )
            except (TypeError, pydantic.ValidationError) as err:
                if self._strict_mode:
                    raise ValueError(
                        "Encountered problem when executing prompt. Ensure your few-shot examples and document "
                        "chunks contain sensible information."
                    ) from err
                else:
                    yield None
