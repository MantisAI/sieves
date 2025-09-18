"""DSPy engine integration for Sieves."""

import asyncio
import enum
import itertools
import sys
from collections.abc import Iterable
from typing import Any, TypeAlias, override

import dspy
import pydantic

from sieves.engines.core import Executable, InternalEngine

PromptSignature: TypeAlias = dspy.Signature | dspy.Module
Model: TypeAlias = dspy.LM | dspy.BaseLM
Result: TypeAlias = dspy.Prediction


class InferenceMode(enum.Enum):
    """Available inference modes.

    See https://dspy.ai/#__tabbed_2_6 for more information and examples.
    """

    # Default inference mode.
    predict = dspy.Predict
    # CoT-style inference.
    chain_of_thought = dspy.ChainOfThought
    # Agentic, i.e. with tool use.
    react = dspy.ReAct
    # For multi-stage pipelines within a task. This is handled differently than the other supported modules: dspy.Module
    # serves as both the signature as well as the inference generator.
    module = dspy.Module


class DSPy(InternalEngine[PromptSignature, Result, Model, InferenceMode]):
    """Engine for DSPy."""

    def __init__(
        self,
        model: Model,
        config_kwargs: dict[str, Any],
        init_kwargs: dict[str, Any],
        inference_kwargs: dict[str, Any],
        strict_mode: bool,
        batch_size: int,
    ):
        """Initialize engine.

        :param model: Model to run. Note: DSPy only runs with APIs. If you want to run a model locally from v2.5
            onwards, serve it with OLlama - see here: # https://dspy.ai/learn/programming/language_models/?h=models#__tabbed_1_5.
            In a nutshell:
            > curl -fsSL https://ollama.ai/install.sh | sh
            > ollama run MODEL_ID
            > `model = dspy.LM(MODEL_ID, api_base='http://localhost:11434', api_key='')`
        :param config_kwargs: Optional kwargs supplied to dspy.configure().
        :param init_kwargs: Optional kwargs to supply to engine executable at init time.
        :param inference_kwargs: Optional kwargs to supply to engine executable at inference time.
        :param strict_mode: If True, exception is raised if prompt response can't be parsed correctly.
        :param batch_size: Batch size in processing prompts. -1 will batch all documents in one go. Not all engines
            support batching.
        """
        super().__init__(model, init_kwargs, inference_kwargs, strict_mode, batch_size)
        config_kwargs = {"max_tokens": DSPy._MAX_TOKENS} | (config_kwargs or {})
        dspy.configure(lm=model, **config_kwargs)

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
        prompt_template: str | None,  # noqa: UP007
        prompt_signature: type[PromptSignature] | PromptSignature,
        fewshot_examples: Iterable[pydantic.BaseModel] = tuple(),
    ) -> Executable[Result | None]:
        # Note: prompt_template is ignored here, as DSPy doesn't use it directly (only prompt_signature_description).
        assert isinstance(prompt_signature, type)

        # Handled differently than the other supported modules: dspy.Module serves as both the signature as well as
        # the inference generator.
        if inference_mode == InferenceMode.module:
            assert isinstance(prompt_signature, dspy.Module), ValueError(
                "In inference mode 'module' the provided prompt signature has to be of type dspy.Module."
            )
            generator = inference_mode.value(**self._init_kwargs)
        else:
            assert issubclass(prompt_signature, dspy.Signature)
            generator = inference_mode.value(signature=prompt_signature, **self._init_kwargs)

        def execute(values: Iterable[dict[str, Any]]) -> Iterable[Result | None]:
            # Compile predictor with few-shot examples.
            fewshot_examples_dicts = DSPy._convert_fewshot_examples(fewshot_examples)
            generator_fewshot: dspy.Module | None = None
            if len(fewshot_examples_dicts):
                examples = [dspy.Example(**fs_example) for fs_example in fewshot_examples_dicts]
                generator_fewshot = dspy.LabeledFewShot(k=5).compile(student=generator, trainset=examples)
            generator_async = dspy.asyncify(generator_fewshot or generator)

            batch_size = self._batch_size if self._batch_size != -1 else sys.maxsize
            # Ensure values are read as generator for standardized batch handling (otherwise we'd have to use
            # different batch handling depending on whether lists/tuples or generators are used).
            values = (v for v in values)

            while batch := [vals for vals in itertools.islice(values, batch_size)]:
                if len(batch) == 0:
                    break

                try:
                    calls = [generator_async(**doc_values, **self._inference_kwargs) for doc_values in batch]
                    yield from asyncio.run(self._execute_async_calls(calls))

                except ValueError as err:
                    if self._strict_mode:
                        raise ValueError(
                            "Encountered problem when executing prompt. Ensure your few-shot examples and document "
                            "chunks contain sensible information."
                        ) from err
                    else:
                        yield from [None] * len(batch)

        return execute
