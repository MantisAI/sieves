from collections.abc import Iterable
from enum import StrEnum
from typing import Any, TypeAlias

import pydantic
import pydantic_core
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from sieves.engines.core import Executable, PydanticEngine

PromptSignature: TypeAlias = pydantic.BaseModel | list[str] | str
Model: TypeAlias = LLM
Result: TypeAlias = pydantic.BaseModel | str


class InferenceMode(StrEnum):
    """Available inference modes."""

    json = "json"
    choice = "choice"
    regex = "regex"
    grammar = "grammar"


class VLLM(PydanticEngine[PromptSignature, Result, Model, InferenceMode]):
    """vLLM engine.
    Note: if you don't have a GPU, you have to install vLLM from source. Follow the instructions given in
    https://docs.vllm.ai/en/v0.6.1/getting_started/cpu-installation.html.
    """

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

        def execute(values: Iterable[dict[str, Any]]) -> Iterable[Result | None]:
            """Execute prompts with engine for given values.
            :param values: Values to inject into prompts.
            :return Iterable[Result | None]: Results for prompts. Results are None if corresponding prompt failed.
            """
            # If Pydantic model: convert into JSON schema.
            converted_decoding_params: type[PromptSignature] | PromptSignature | dict[str, Any] = prompt_signature
            if inference_mode == InferenceMode.json:
                assert issubclass(prompt_signature, pydantic.BaseModel)  # type: ignore[arg-type]
                assert hasattr(prompt_signature, "model_json_schema")
                converted_decoding_params = prompt_signature.model_json_schema()

            guided_decoding_params = GuidedDecodingParams(**{inference_mode.value: converted_decoding_params})
            sampling_params = SamplingParams(
                guided_decoding=guided_decoding_params, **({"max_tokens": VLLM._MAX_TOKENS} | self._init_kwargs)
            )

            def generate(prompts: list[str]) -> Iterable[Result]:
                results = self._model.generate(
                    prompts=prompts, sampling_params=sampling_params, **({"use_tqdm": False} | self._inference_kwargs)
                )

                for result in results:
                    match inference_mode:
                        case InferenceMode.json:
                            assert issubclass(prompt_signature, pydantic.BaseModel)  # type: ignore[arg-type]
                            assert hasattr(prompt_signature, "model_validate")
                            result_as_json = pydantic_core.from_json(result.outputs[0].text, allow_partial=True)
                            result_structured = prompt_signature.model_validate(result_as_json)
                            yield result_structured

                        case InferenceMode.choice:
                            assert isinstance(prompt_signature, list)
                            result_as_json = pydantic_core.from_json(result.outputs[0].text, allow_partial=True)
                            yield result_as_json

                        case _:
                            yield result.outputs[0].text

            yield from self._infer(
                generate,
                template,
                values,
                fewshot_examples,
            )

        return execute
