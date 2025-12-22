"""GLiNER2 model wrapper wrapper built on top of GLiNER2 multi‑task pipelines."""

import enum
import warnings
from collections.abc import Sequence
from typing import Any, override

import gliner2
import jinja2
import pydantic

from sieves.model_wrappers.core import Executable, ModelWrapper
from sieves.model_wrappers.types import TokenUsage

PromptSignature = gliner2.inference.engine.Schema | gliner2.inference.engine.StructureBuilder
Model = gliner2.GLiNER2
Result = dict[str, str | list[str | dict[str, Any]]]


class InferenceMode(enum.Enum):
    """Available inference modes."""

    classification = 1
    entities = 2
    structure = 3


class GliNER(ModelWrapper[PromptSignature, Result, Model, InferenceMode]):
    """ModelWrapper adapter for GLiNER2."""

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
            warnings.warn(f"Few-shot examples are not supported by model type {cls_name}.")

        # Overwrite prompt default template, if template specified. Note that this is a static prompt and GliNER doesn't
        # do few-shotting, so we don't inject anything into the template.
        if prompt_template:
            self._model.prompt = jinja2.Template(prompt_template).render()

        def execute(values: Sequence[dict[str, Any]]) -> Sequence[tuple[Result, Any, TokenUsage]]:
            """Execute prompts with model wrapper for given values.

            :param values: Values to inject into prompts.
            :return: Sequence of tuples containing results, raw outputs, and token usage.
            """
            results = self._model.batch_extract(
                texts=[val["text"] for val in values],
                schemas=prompt_signature,
                **(
                    {"batch_size": len(values)}
                    | self._inference_kwargs
                    | {"include_confidence": True, "include_spans": True}
                ),
            )

            # Estimate token usage if tokenizer is available.
            tokenizer = self._get_tokenizer()

            final_results: list[tuple[Result, Any, TokenUsage]] = []
            # Estimate token counts.
            for val, res in zip(values, results):
                usage = TokenUsage(
                    input_tokens=self._count_tokens(val["text"], tokenizer),
                    output_tokens=self._count_tokens(str(res), tokenizer),
                )

                final_results.append((res, res, usage))
            return final_results

        return execute

    @override
    def _get_tokenizer(self) -> Any | None:
        # GLiNER2 models usually have a tokenizer accessible through the processor.
        tokenizer = getattr(self._model, "tokenizer", None)
        if not tokenizer and hasattr(self._model, "processor"):
            tokenizer = getattr(self._model.processor, "tokenizer", None)

        return tokenizer
