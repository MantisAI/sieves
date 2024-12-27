import inspect
from typing import Any, Callable, Iterable

import outlines
import pydantic

from sieves.engines import Engine

Executable = Callable[[Iterable[dict[str, Any]]], Iterable[pydantic.BaseModel]]


class Outlines(Engine[outlines.prompts.Prompt, pydantic.BaseModel, Executable]):
    @classmethod
    def convert_prompt_template(cls, prompt_template: str) -> str:
        # Nothing to do, outlines works with Jinja 2 templates.
        return prompt_template

    @classmethod
    def render_prompts(
        cls, prompt_template: str, values: Iterable[dict[str, Any]]
    ) -> Iterable[outlines.prompts.Prompt]:
        for curr_values in values:
            # Outlines expect Callable/Signature object to create a bond, so we create a Signature dynamically.
            signature = cls._create_fn_signature(
                ((k, inspect.Parameter.POSITIONAL_OR_KEYWORD, Any, None) for k, v in curr_values.items())
            )
            yield outlines.prompts.Prompt(prompt_template, signature)

    @outlines.prompt
    def build_executable(
        self, prompt_template: outlines.prompts.Prompt, prompt_signature: pydantic.BaseModel
    ) -> Executable:
        executable = outlines.Function(prompt_template, prompt_signature, self._model_id)

        def execute(values: Iterable[dict[str, Any]]) -> Iterable[pydantic.BaseModel]:
            return (executable(doc_values) for doc_values in values)

        return execute

    @staticmethod
    def _create_fn_signature(*parameters: Iterable[tuple[str, inspect.Parameter.kind, type, Any]]) -> inspect.Signature:
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
