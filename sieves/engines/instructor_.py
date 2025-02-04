import enum
import warnings
from collections.abc import Iterable
from typing import Any, TypeAlias

import instructor
import pydantic

from sieves.engines.core import Executable, PydanticEngine


class Model(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    name: str
    client: instructor.Instructor | instructor.AsyncInstructor


PromptSignature: TypeAlias = pydantic.BaseModel
Result: TypeAlias = pydantic.BaseModel


class InferenceMode(enum.Enum):
    chat = "chat"


class Instructor(PydanticEngine[PromptSignature, Result, Model, InferenceMode]):
    def _validate_batch_size(self, batch_size: int) -> int:
        if not isinstance(self._model.client, instructor.AsyncInstructor):
            if batch_size != 1:
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
            match inference_mode:
                case InferenceMode.chat:

                    def generate(prompts: list[str]) -> Iterable[Result]:
                        # todo generate is only for _one_ prompt. this doesn't a batched pattern -> probably best to:
                        #   - extend PydanticEngine.infer() to support batch generators
                        #   - specify so via a batch_size arg to infer() that defaults to 1
                        #   - implement async calls for a list in here, but batching in infer()
                        #       -> reasoning: batching is a general mechanism, but it's far from guaranteed that other
                        #          engines will batch the same way (i.e. via async calls - could be via other
                        #          inference server optimizations). if it becomes clear that this is the case, we can
                        #          still lift the async call mechanism up into PydanticEngine.
                        raise NotImplementedError
                        # if isinstance(self._model, instructor.AsyncInstructor):
                        #     calls: list[Coroutine] = [
                        #         self._model.client.chat.completions.create(
                        #             model=self._model.name,
                        #             messages=[{"role": "user", "content": prompt}],
                        #             response_model=prompt_signature,
                        #             **inference_kwargs,
                        #         )
                        #     ]
                        #
                        #     for i in range(0, len(calls), batch_size):
                        #         results_batch = [
                        #             res for res in asyncio.run(_download(calls[i : i + batch_size])) if res
                        #         ]
                        #
                        #     asyncio.run(calls)
                        #     # todo implement async calling with batch size
                        #     pass
                        # else:
                        #     result = self._model.client.chat.completions.create(
                        #         model=self._model.name,
                        #         messages=[{"role": "user", "content": prompt}],
                        #         response_model=prompt_signature,
                        #         **inference_kwargs,
                        #     )
                        #     assert isinstance(result, prompt_signature)
                        #     return result

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
