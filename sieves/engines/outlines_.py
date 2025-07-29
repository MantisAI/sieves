import enum
from collections.abc import Iterable
from typing import Any, TypeAlias

import outlines
import pydantic
from outlines.models import AsyncBlackBoxModel, BlackBoxModel, SteerableModel

from sieves.engines.core import Executable, PydanticEngine

PromptSignature: TypeAlias = (
    pydantic.BaseModel | list[str] | str | outlines.types.Choice | outlines.types.Regex | outlines.types.JsonSchema
)
Model: TypeAlias = AsyncBlackBoxModel | BlackBoxModel | SteerableModel
Result: TypeAlias = pydantic.BaseModel | str


class InferenceMode(enum.Enum):
    """Available inference modes.
    Note: generator functions are wrapped in tuples, as otherwise the Enum instance seems to be replaced by the function
    itself - not sure why that happens. Should take another look at this.
    """

    # For normal text output, i.e. no structured generation.
    text = "text"
    # For limited set of choices, e.g. classification.
    choice = "choice"
    # Regex-conforming output.
    regex = "regex"
    # Output conforming to Pydantic models.
    json = "json"


class Outlines(PydanticEngine[PromptSignature, Result, Model, InferenceMode]):
    @property
    def inference_modes(self) -> type[InferenceMode]:
        return InferenceMode

    def build_executable(
        self,
        inference_mode: InferenceMode,
        prompt_template: str | None,  # noqa: UP007
        prompt_signature: type[PromptSignature] | PromptSignature,
        fewshot_examples: Iterable[pydantic.BaseModel] = (),
    ) -> Executable[Result | None]:
        template = self._create_template(prompt_template)

        generator = outlines.Generator(self._model, output_type=prompt_signature, **self._init_kwargs)

        def execute(values: Iterable[dict[str, Any]]) -> Iterable[Result | None]:
            """Execute prompts with engine for given values.
            :param values: Values to inject into prompts.
            :return Iterable[Result | None]: Results for prompts. Results are None if corresponding prompt failed.
            """

            def generate(prompts: list[str]) -> Iterable[Result]:
                yield from generator(prompts, **self._inference_kwargs)

            yield from self._infer(
                generate,
                template,
                values,
                fewshot_examples,
            )

        return execute
