import enum
import warnings
from collections.abc import Iterable
from typing import Any, TypeAlias

import gliner.multitask.base
import pydantic

from sieves.engines.core import Engine, Executable

PromptSignature: TypeAlias = list[str]
Model: TypeAlias = gliner.multitask.base.GLiNERBasePipeline
Result: TypeAlias = list[dict[str, str | float]]


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
        prompt_signature: PromptSignature,
        fewshot_examples: Iterable[pydantic.BaseModel] = (),
    ) -> Executable[Result]:
        cls_name = self.__class__.__name__
        fewshot_examples = list(fewshot_examples)
        if prompt_template:
            warnings.warn(f"prompt_template is ignored by engine {cls_name}.")
        if len(fewshot_examples):
            warnings.warn(f"Few-shot examples are not supported by engine {cls_name}.")

        def execute(values: Iterable[dict[str, Any]]) -> Iterable[Result]:
            texts = [dv["text"] for dv in values]

            match inference_mode:
                case InferenceMode.classification:
                    result = self._model(
                        texts, classes=prompt_signature, **({"multi_label": True} | self._inference_kwargs)
                    )
                case _:
                    raise ValueError(f"Inference mode {inference_mode} not supported by {cls_name} engine.")

            assert isinstance(result, Iterable)
            return result

        return execute
