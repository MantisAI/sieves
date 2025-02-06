import asyncio
import enum
import warnings
from collections.abc import Iterable
from typing import Any, Literal, TypeAlias

import ollama
import pydantic

from sieves.engines.core import Executable, PydanticEngine


class Model(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    name: str
    client_mode: Literal["sync", "async"]
    client_config: dict[str, Any] = pydantic.Field(default_factory=dict)
    host: str


PromptSignature: TypeAlias = pydantic.BaseModel
Result: TypeAlias = pydantic.BaseModel


class InferenceMode(enum.Enum):
    chat = "chat"


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
        super().__init__(
            model=model,
            init_kwargs=init_kwargs,
            inference_kwargs=inference_kwargs,
            strict_mode=strict_mode,
            batch_size=batch_size,
        )
        # Async client will be initialized for every prompt batch to sidestep an asyncio event loop issue.
        self._client: ollama.Client | ollama.AsyncClient | None = (
            ollama.Client(host=model.host, **model.client_config) if model.client_mode == "sync" else None
        )

    def _validate_batch_size(self, batch_size: int) -> int:
        if not self._model.client_mode == "sync" and batch_size != 1:
            warnings.warn(
                f"`batch_size` is forced to 1 when {self.__class__.__name__} engine is run with `Instructor`, as "
                f"it runs a synchronous workflow."
            )
            batch_size = 1

        return batch_size

    @property
    def inference_modes(self) -> type[InferenceMode]:
        return InferenceMode

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
            """
            match inference_mode:
                case InferenceMode.chat:

                    def generate(prompts: list[str]) -> Iterable[Result]:
                        if self._model.client_mode == "async":
                            self._client = ollama.AsyncClient(host=self._model.host, **self._model.client_config)
                        assert self._client

                        responses = [
                            self._client.chat(
                                messages=[{"role": "user", "content": prompt}],
                                model=self._model.name,
                                format=prompt_signature.model_json_schema(),
                                **self._inference_kwargs,
                            )
                            for prompt in prompts
                        ]

                        # For async client: responses are coroutines waiting to be executed.
                        if self._model.client_mode == "async":
                            responses = asyncio.run(self._execute_async_calls(responses))

                        try:
                            for res in responses:
                                yield prompt_signature.model_validate_json(res.message.content)
                        except pydantic.ValidationError as ex:
                            raise pydantic.ValidationError(
                                f"Encountered problem in parsing {cls_name} output. Double-check your prompts and "
                                f"examples."
                            ) from ex

                case _:
                    raise ValueError(f"Inference mode {inference_mode} not supported by {cls_name} engine.")

            yield from self._infer(generate, template, values, fewshot_examples)

        return execute
