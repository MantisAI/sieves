import enum
import itertools
import sys
import warnings
from collections.abc import Iterable
from typing import Any, TypeAlias

import gliner.multitask.base
import jinja2
import pydantic

from sieves.engines.core import Engine, Executable

PromptSignature: TypeAlias = list[str]
Model: TypeAlias = gliner.multitask.base.GLiNERBasePipeline
Result: TypeAlias = list[dict[str, str | float]] | str


class InferenceMode(enum.Enum):
    """Available inference modes."""

    ner = 0
    classification = 1
    question_answering = 2
    information_extraction = 3
    summarization = 4


class GliX(Engine[PromptSignature, Result, Model, InferenceMode]):
    @property
    def inference_modes(self) -> type[InferenceMode]:
        return InferenceMode

    @property
    def supports_few_shotting(self) -> bool:
        return False

    def build_executable(
        self,
        inference_mode: InferenceMode,
        prompt_template: str | None,
        prompt_signature: type[PromptSignature] | PromptSignature,
        fewshot_examples: Iterable[pydantic.BaseModel] = (),
    ) -> Executable[Result]:
        assert isinstance(prompt_signature, list)
        cls_name = self.__class__.__name__
        if len(list(fewshot_examples)):
            warnings.warn(f"Few-shot examples are not supported by engine {cls_name}.")

        # Overwrite prompt default template, if template specified. Note that this is a static prompt and GliX doesn't
        # do few-shotting, so we don't inject anything into the template.
        if prompt_template:
            self._model.prompt = jinja2.Template(prompt_template).render()

        def execute(values: Iterable[dict[str, Any]]) -> Iterable[Result]:
            """Execute prompts with engine for given values.
            :param values: Values to inject into prompts.
            :return Iterable[Result]: Results for prompts.
            """
            try:
                params = {
                    InferenceMode.classification: {"classes": prompt_signature, "multi_label": True},
                    InferenceMode.question_answering: {"questions": prompt_signature},
                    InferenceMode.summarization: {},
                }[inference_mode]
            except KeyError:
                raise ValueError(f"Inference mode {inference_mode} not supported by {cls_name} engine.")

            batch_size = self._batch_size if self._batch_size != -1 else sys.maxsize
            # Ensure values are read as generator for standardized batch handling (otherwise we'd have to use
            # different batch handling depending on whether lists/tuples or generators are used).
            values = (v for v in values)

            while batch := [vals["text"] for vals in itertools.islice(values, batch_size)]:
                if len(batch) == 0:
                    break

                assert isinstance(params, dict)
                yield from self._model(batch, **(params | self._inference_kwargs))

        return execute
