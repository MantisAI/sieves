import enum
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias

import jinja2
import pydantic
import transformers

from sieves.engines.core import Engine, Executable

PromptSignature: TypeAlias = list[str]
Model: TypeAlias = transformers.Pipeline
Result: TypeAlias = dict[str, list[str] | list[float]]


class InferenceMode(enum.Enum):
    """Available inference modes."""

    zeroshot_cls = 0


class HuggingFace(Engine[PromptSignature, Result, Model, InferenceMode]):
    @property
    def inference_modes(self) -> type[InferenceMode]:
        return InferenceMode

    @property
    def supports_few_shotting(self) -> bool:
        return True

    def build_executable(
        self,
        inference_mode: InferenceMode,
        prompt_template: str | None,
        prompt_signature: PromptSignature,
        fewshot_examples: Iterable[pydantic.BaseModel] = (),
    ) -> Executable[Result]:
        cls_name = self.__class__.__name__
        assert prompt_template, ValueError(f"prompt_template has to be provided to {cls_name} engine by task.")

        # Convert few-shot examples.
        fs_examples: list[dict[str, Any]] = []
        for fs_example in fewshot_examples:
            fs_examples.append(fs_example.model_dump(serialize_as_any=True))

        # Render template with few-shot examples. Note that we don't use extracted document values here, as HF zero-shot
        # pipelines only support one hypothesis template per call - and we want to batch, so our hypothesis template
        # will be document-invariant.
        template = jinja2.Template(prompt_template).render(**({"examples": fs_examples} if len(fs_examples) else {}))

        def execute(values: Iterable[dict[str, Any]]) -> Iterable[Result]:
            generator: Callable[[Iterable[str]], Iterable[Result]]

            match inference_mode:
                case InferenceMode.zeroshot_cls:

                    def generate(texts: Iterable[str]) -> Iterable[Result]:
                        result = self._model(
                            texts,
                            prompt_signature,
                            # Render hypothesis template with everything but text.
                            hypothesis_template=template,
                            multi_label=True,
                            **self._inference_kwargs,
                        )
                        assert isinstance(result, Iterable)
                        return result

                    generator = generate
                case _:
                    raise ValueError(f"Inference mode {inference_mode} not supported by {cls_name} engine.")

            return generator([doc_values["text"] for doc_values in values])

        return execute
