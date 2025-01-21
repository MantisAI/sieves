import enum
from collections.abc import Iterable
from typing import Any, TypeAlias

import langchain_core.language_models
import pydantic

from sieves.engines.core import Executable, TemplateBasedEngine

Model: TypeAlias = langchain_core.language_models.BaseChatModel
PromptSignature: TypeAlias = type[pydantic.BaseModel]
Result: TypeAlias = pydantic.BaseModel


class InferenceMode(enum.Enum):
    structured_output = "structured_output"


class LangChain(TemplateBasedEngine[PromptSignature, Result, Model, InferenceMode]):
    """Engine for LangChain."""

    @property
    def inference_modes(self) -> type[InferenceMode]:
        return InferenceMode

    @property
    def supports_few_shotting(self) -> bool:
        return True

    def build_executable(
        self,
        inference_mode: InferenceMode,
        prompt_template: str | None,  # noqa: UP007
        prompt_signature: PromptSignature,
        fewshot_examples: Iterable[pydantic.BaseModel] = tuple(),
    ) -> Executable[Result | None]:
        cls_name = self.__class__.__name__
        template = self._create_template(prompt_template)

        def execute(values: Iterable[dict[str, Any]]) -> Iterable[Result | None]:
            match inference_mode:
                case InferenceMode.structured_output:
                    model = self._model.with_structured_output(prompt_signature)

                    def generate(prompt: str, **inference_kwargs: dict[str, Any]) -> Result:
                        try:
                            result = model.invoke(prompt, **inference_kwargs)
                            assert isinstance(result, Result)
                            return result
                        except pydantic.ValidationError as ex:
                            raise pydantic.ValidationError(
                                "Encountered problem in parsing Ollama output. Double-check your prompts and examples."
                            ) from ex

                    generator = generate
                case _:
                    raise ValueError(f"Inference mode {inference_mode} not supported by {cls_name} engine.")

            return self._infer(
                generator,
                template,
                values,
                fewshot_examples,
            )

        return execute
