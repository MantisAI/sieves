"""vLLM engine wrapper enabling guided decoding for structured outputs."""

import re
from collections.abc import Iterable
from enum import StrEnum
from typing import Any, TypeAlias, override

import json_repair
import pydantic
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

    @override
    @property
    def inference_modes(self) -> type[InferenceMode]:
        return InferenceMode

    @override
    def build_executable(
        self,
        inference_mode: InferenceMode,
        prompt_template: str | None,  # noqa: UP007
        prompt_signature: type[PromptSignature] | PromptSignature,
        fewshot_examples: Iterable[pydantic.BaseModel] = (),
    ) -> Executable[Result | None]:
        template = self._create_template(prompt_template)

        # If Pydantic model: convert into JSON schema.
        converted_decoding_params: type[PromptSignature] | PromptSignature | dict[str, Any] = prompt_signature
        if inference_mode == InferenceMode.json:
            assert issubclass(prompt_signature, pydantic.BaseModel)  # type: ignore[arg-type]
            assert hasattr(prompt_signature, "model_json_schema")
            converted_decoding_params = prompt_signature.model_json_schema()

        # Configure decoding params to encourage correct formatting.
        guided_decoding_params = GuidedDecodingParams(**{inference_mode.value: converted_decoding_params})
        sampling_params = SamplingParams(
            guided_decoding=guided_decoding_params,
            **({"max_tokens": VLLM._MAX_TOKENS, "temperature": 0} | self._init_kwargs),
        )

        def execute(values: Iterable[dict[str, Any]]) -> Iterable[Result | None]:
            """Execute prompts with engine for given values.

            :param values: Values to inject into prompts.
            :return Iterable[Result | None]: Results for prompts. Results are None if corresponding prompt failed.
            """

            def generate(prompts: list[str]) -> Iterable[Result]:
                results = self._model.generate(
                    prompts=prompts, sampling_params=sampling_params, **({"use_tqdm": False} | self._inference_kwargs)
                )

                for result in results:
                    # Sanitize output by removing invalid control characters.
                    # This pattern matches all C0 controls except tab (\x09), LF (\x0A) and CR (\x0D)
                    control_chars = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]+")
                    sanitized_result = control_chars.sub("", result.outputs[0].text)

                    match inference_mode:
                        case InferenceMode.json:
                            assert issubclass(prompt_signature, pydantic.BaseModel)  # type: ignore[arg-type]
                            assert hasattr(prompt_signature, "model_validate")
                            result_as_json = json_repair.repair_json(sanitized_result, ensure_ascii=False)
                            result_structured = prompt_signature.model_validate(result_as_json)
                            yield result_structured

                        case InferenceMode.choice:
                            assert isinstance(prompt_signature, list)
                            yield sanitized_result

                        case _:
                            yield sanitized_result

            yield from self._infer(
                generate,
                template,
                values,
                fewshot_examples,
            )

        return execute
