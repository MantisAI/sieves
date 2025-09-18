"""Ollama engine wrapper for structured generation against a running server."""

import asyncio
import enum
from collections.abc import Iterable
from typing import Any, TypeAlias, override

import httpx
import ollama
import pydantic

from sieves.engines.core import Executable, PydanticEngine


class Model(pydantic.BaseModel):
    """Configuration for connecting to an Ollama server."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    name: str
    host: str
    max_retries: int = pydantic.Field(default=5)
    timeout: int = pydantic.Field(default=10)
    client_config: dict[str, Any] = pydantic.Field(default_factory=dict)


PromptSignature: TypeAlias = pydantic.BaseModel
Result: TypeAlias = pydantic.BaseModel


class InferenceMode(enum.Enum):
    """Available inference modes."""

    structured = "structured"


class Ollama(PydanticEngine[PromptSignature, Result, Model, InferenceMode]):
    """Engine for Ollama.

    Make sure a Ollama server is running.
            In a nutshell:
            > curl -fsSL https://ollama.ai/install.sh | sh
            > ollama serve (or ollama run MODEL_ID)
    """

    def __init__(
        self,
        model: Model,
        init_kwargs: dict[str, Any] | None = None,
        inference_kwargs: dict[str, Any] | None = None,
        strict_mode: bool = False,
        batch_size: int = -1,
    ):
        """Initialize Ollama engine with client and retry configuration."""
        super().__init__(
            model=model,
            init_kwargs=init_kwargs,
            inference_kwargs=inference_kwargs,
            strict_mode=strict_mode,
            batch_size=batch_size,
        )

        # Async client will be initialized for every prompt batch to sidestep an asyncio event loop issue.
        self._client = ollama.AsyncClient(host=self._model.host, **({"timeout": 30} | self._model.client_config))

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
        fewshot_examples: Iterable[pydantic.BaseModel] = tuple(),
    ) -> Executable[Result | None]:
        assert isinstance(prompt_signature, type)
        cls_name = self.__class__.__name__
        template = self._create_template(prompt_template)

        def execute(values: Iterable[dict[str, Any]]) -> Iterable[Result | None]:
            """Execute prompts with engine for given values.

            :param values: Values to inject into prompts.
            :return: Results for prompts. Results are None if corresponding prompt failed.
            :raises: pydantic.ValidationError is response can't be parsed.
            :raises: httpx.ReadTimeout if request times out.
            """
            match inference_mode:
                case InferenceMode.structured:

                    def generate(prompts: list[str]) -> Iterable[Result]:
                        responses: list[Any] | None = None
                        n_tries = 0

                        while responses is None:
                            calls = [
                                self._client.chat(
                                    messages=[{"role": "user", "content": prompt}],
                                    model=self._model.name,
                                    format=prompt_signature.model_json_schema(),
                                    **self._inference_kwargs,
                                )
                                for prompt in prompts
                            ]

                            try:
                                responses = asyncio.run(self._execute_async_calls(calls))

                                try:
                                    for res in responses:
                                        yield prompt_signature.model_validate_json(res.message.content)
                                except pydantic.ValidationError as ex:
                                    raise pydantic.ValidationError(
                                        f"Encountered problem in parsing {cls_name} output. Double-check your "
                                        f"prompts and examples."
                                    ) from ex

                            except (RuntimeError, httpx.ReadTimeout) as err:
                                n_tries += 1
                                if n_tries > self._model.max_retries:
                                    raise TimeoutError("Hit timeout limit with Ollama.") from err

                                self._client = ollama.AsyncClient(
                                    host=self._model.host,
                                    **({"timeout": self._model.timeout} | self._model.client_config),
                                )

                case _:
                    raise ValueError(f"Inference mode {inference_mode} not supported by {cls_name} engine.")

            yield from self._infer(generate, template, values, fewshot_examples)

        return execute
