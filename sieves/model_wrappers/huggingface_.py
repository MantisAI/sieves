"""Hugging Face transformers zero-shot classification pipeline model wrapper."""

import enum
from collections.abc import Sequence
from typing import Any, override

import jinja2
import pydantic
import transformers

from sieves.model_wrappers.core import Executable, ModelWrapper
from sieves.model_wrappers.types import TokenUsage

PromptSignature = list[str]
Model = transformers.Pipeline
Result = dict[str, list[str] | list[float]]


class InferenceMode(enum.Enum):
    """Available inference modes."""

    zeroshot_cls = 0


class HuggingFace(ModelWrapper[PromptSignature, Result, Model, InferenceMode]):
    """ModelWrapper adapter around ``transformers.Pipeline`` for zero‑shot tasks."""

    @override
    @property
    def inference_modes(self) -> type[InferenceMode]:
        return InferenceMode

    @override
    @property
    def supports_few_shotting(self) -> bool:
        return True

    @override
    def build_executable(
        self,
        inference_mode: InferenceMode,
        prompt_template: str | None,
        prompt_signature: type[PromptSignature] | PromptSignature,
        fewshot_examples: Sequence[pydantic.BaseModel] = (),
    ) -> Executable[Result | None]:
        cls_name = self.__class__.__name__
        assert prompt_template, ValueError(f"prompt_template has to be provided to {cls_name} model wrapper by task.")
        assert isinstance(prompt_signature, list)

        # Render template with few-shot examples. Note that we don't use extracted document values here, as HF zero-shot
        # pipelines only support one hypothesis template per _call - and we want to batch, so our hypothesis template
        # will be document-invariant.
        fewshot_examples_dict = HuggingFace.convert_fewshot_examples(fewshot_examples)
        # Render hypothesis template with everything but text.
        template = jinja2.Template(prompt_template).render(**({"examples": fewshot_examples_dict}))

        def execute(values: Sequence[dict[str, Any]]) -> Sequence[tuple[Result | None, Any, TokenUsage]]:
            """Execute prompts with model wrapper for given values.

            :param values: Values to inject into prompts.
            :return: Sequence of tuples containing results, raw outputs, and token usage.
            """
            match inference_mode:
                case InferenceMode.zeroshot_cls:
                    results = self._model(
                        sequences=[doc_values["text"] for doc_values in values],
                        candidate_labels=prompt_signature,
                        hypothesis_template=template,
                        multi_label=True,
                        **self._inference_kwargs,
                    )

                    # Estimate token usage if tokenizer is available.
                    tokenizer = self._get_tokenizer()

                    final_results: list[tuple[Result, Any, TokenUsage]] = []
                    for doc_values, res in zip(values, results):
                        usage = TokenUsage(
                            input_tokens=self._count_tokens(doc_values["text"], tokenizer),
                            # For classification, we estimate output tokens based on the labels.
                            output_tokens=self._count_tokens(" ".join(res["labels"]), tokenizer),
                        )

                        final_results.append((res, res, usage))
                    return final_results

                case _:
                    raise ValueError(f"Inference mode {inference_mode} not supported by {cls_name} model wrapper.")

        return execute

    @override
    def _get_tokenizer(self) -> Any | None:
        return getattr(self._model, "tokenizer", None)
