import inspect
from typing import Any, Iterable

import outlines
import pydantic
from engines.core import IExecutable
from typing_extensions import TypeAlias

from sieves.engines import Engine

# Define the parameter kind type properly
ParameterKind: TypeAlias = inspect._ParameterKind


class Executable(IExecutable[pydantic.BaseModel]):  # type: ignore[misc]
    def __call__(self, values: Iterable[dict[str, Any]]) -> Iterable[pydantic.BaseModel]:  # type: ignore[empty-body]
        ...


class Outlines(Engine[outlines.prompts.Prompt, pydantic.BaseModel, Executable]):
    @classmethod
    def convert_prompt_template(cls, prompt_template: str, variable_names: tuple[str] = ()) -> outlines.prompts.Prompt:  # type: ignore[assignment]
        # Outlines expect Callable/Signature object to create a bind, so we create a Signature dynamically.
        signature = cls._create_fn_signature(
            (var_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, Any, None) for var_name in variable_names
        )
        return outlines.prompts.Prompt(prompt_template, signature)

    def build_executable(
        self, prompt_template: outlines.prompts.Prompt, prompt_signature: pydantic.BaseModel
    ) -> Executable:
        executable = outlines.Function(prompt_template, prompt_signature, self._model_id)

        class _Executable(Executable):
            def execute(self, values: Iterable[dict[str, Any]]) -> Iterable[pydantic.BaseModel]:
                return (executable(doc_values) for doc_values in values)

        return _Executable()

    @staticmethod
    def _create_fn_signature(parameters: Iterable[tuple[str, ParameterKind, Any, Any]]) -> inspect.Signature:
        """
        Create an inspect.Signature object.
        :param parameters: Tuples of (name, kind, annotation, default).
        :returns: Instance of inspect.Signature.
        """
        params: list[inspect.Parameter] = []
        for name, kind, annotation, default in parameters:
            param = inspect.Parameter(
                name=name,
                kind=kind,
                annotation=annotation,
                default=default if default is not None else inspect.Parameter.empty,
            )
            params.append(param)

        return inspect.Signature(params)
