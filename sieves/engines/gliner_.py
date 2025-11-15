"""GLiNER2 engine wrapper built on top of GLiNER2 multi‑task pipelines."""

import enum
import warnings
from collections.abc import Iterable, Sequence
from typing import Any, override

import gliner2
import jinja2
import pydantic

from sieves.engines.core import Engine, Executable

PromptSignature = gliner2.inference.engine.Schema | gliner2.inference.engine.StructureBuilder
Model = gliner2.GLiNER2
Result = dict[str, str | list[str | dict[str, Any]]]


class InferenceMode(enum.Enum):
    """Available inference modes."""

    classification = 1
    entities = 2
    structure = 3


class GliNER(Engine[PromptSignature, Result, Model, InferenceMode]):
    """Engine adapter for GLiNER2."""

    @override
    @property
    def inference_modes(self) -> type[InferenceMode]:
        return InferenceMode

    @override
    @property
    def supports_few_shotting(self) -> bool:
        return False

    @override
    def build_executable(
        self,
        inference_mode: InferenceMode,
        prompt_template: str | None,
        prompt_signature: type[PromptSignature] | PromptSignature,
        fewshot_examples: Sequence[pydantic.BaseModel] = (),
    ) -> Executable[Result]:
        cls_name = self.__class__.__name__
        if len(list(fewshot_examples)):
            warnings.warn(f"Few-shot examples are not supported by engine {cls_name}.")

        # Overwrite prompt default template, if template specified. Note that this is a static prompt and GliNER doesn't
        # do few-shotting, so we don't inject anything into the template.
        if prompt_template:
            self._model.prompt = jinja2.Template(prompt_template).render()

        def execute(values: Sequence[dict[str, Any]]) -> Iterable[Result]:
            """Execute prompts with engine for given values.

            :param values: Values to inject into prompts.
            :return Iterable[Result]: Results for prompts.
            """
            yield from self._model.batch_extract(
                texts=[val["text"] for val in values],
                schemas=prompt_signature,
                **({"batch_size": len(values)} | self._inference_kwargs | {"include_confidence": True}),
            )

        return execute
