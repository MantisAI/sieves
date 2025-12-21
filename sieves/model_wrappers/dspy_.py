"""DSPy model wrapper."""

import asyncio
import enum
from collections.abc import Sequence
from typing import Any, override

import dspy
import litellm
import nest_asyncio
import pydantic

from sieves.model_wrappers.core import Executable, ModelWrapper
from sieves.model_wrappers.types import ModelSettings, TokenUsage

PromptSignature = dspy.Signature | dspy.Module
Model = dspy.LM | dspy.BaseLM
Result = dspy.Prediction


nest_asyncio.apply()


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


class DSPy(ModelWrapper[PromptSignature, Result, Model, InferenceMode]):
    """ModelWrapper for DSPy."""

    def __init__(self, model: Model, model_settings: ModelSettings):
        """Initialize model wrapper.

        :param model: Model to run. Note: DSPy only runs with APIs. If you want to run a model locally from v2.5
            onwards, serve it with OLlama - see here: # https://dspy.ai/learn/programming/language_models/?h=models#__tabbed_1_5.
            In a nutshell:
            > curl -fsSL https://ollama.ai/install.sh | sh
            > ollama run MODEL_ID
            > `model = dspy.LM(MODEL_ID, api_base='http://localhost:11434', api_key='')`
        :param model_settings: Settings including DSPy configuration in `config_kwargs`.
        """
        super().__init__(model, model_settings)
        cfg = model_settings.config_kwargs or {}
        dspy.configure(lm=model, track_usage=True, **cfg)

        # Disable noisy LiteLLM logging.
        dspy.disable_litellm_logging()
        litellm._logging._disable_debugging()

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
        fewshot_examples: Sequence[pydantic.BaseModel] = tuple(),
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

        def execute(values: Sequence[dict[str, Any]]) -> Sequence[tuple[Result | None, Any, TokenUsage]]:
            """Execute structured generation with DSPy.

            :params values: Values to inject into prompts.
            :returns: Sequence of tuples containing results, history entries, and token usage.
            """
            # Compile predictor with few-shot examples.
            fewshot_examples_dicts = DSPy.convert_fewshot_examples(fewshot_examples)
            generator_fewshot: dspy.Module | None = None
            if len(fewshot_examples_dicts):
                examples = [dspy.Example(**fs_example) for fs_example in fewshot_examples_dicts]
                generator_fewshot = dspy.LabeledFewShot(k=len(examples)).compile(student=generator, trainset=examples)

            try:
                gen = generator_fewshot or generator

                async def call_with_meta(**kwargs: Any) -> tuple[Result, Any, TokenUsage]:
                    res = await gen.acall(**kwargs)
                    # Capture the last history entry from the model after call completion.
                    # This works even with concurrency because history is appended upon completion.
                    history_entry = self._model.history[-1]
                    usage = self._extract_usage(res, history_entry)

                    return res, history_entry, usage

                calls = [call_with_meta(**doc_values, **self._inference_kwargs) for doc_values in values]
                return list(asyncio.run(self._execute_async_calls(calls)))

            except Exception as err:
                if self._strict:
                    raise RuntimeError(
                        "Encountered problem when executing prompt. Ensure your few-shot examples and document "
                        "chunks contain sensible information."
                    ) from err
                else:
                    return [(None, None, TokenUsage()) for _ in range(len(values))]

        return execute

    def _extract_usage(self, res: Result, history_entry: Any) -> TokenUsage:
        """Extract token usage from DSPy result and history.

        :param res: DSPy prediction result.
        :param history_entry: Corresponding history entry from the LM.
        :return: Extracted token usage.
        """
        usage = TokenUsage()

        # Try to extract usage from the result object if available (v2.6.16+).
        if hasattr(res, "get_lm_usage"):
            lm_usage = res.get_lm_usage()
            # get_lm_usage() typically returns a dict mapping model names to usage objects.
            if lm_usage and len(lm_usage) > 0:
                # We take the sum of all models if multiple were used.
                usage.input_tokens = sum(u.get("prompt_tokens", 0) for u in lm_usage.values()) or None
                usage.output_tokens = sum(u.get("completion_tokens", 0) for u in lm_usage.values()) or None

        # If usage is still None, try to extract from different possible locations in the history entry.
        # LiteLLM and different DSPy versions store this differently.
        if usage.input_tokens is None:
            raw_usage: Any = None
            if "response" in history_entry and hasattr(history_entry["response"], "usage"):
                raw_usage = history_entry["response"].usage
            elif "usage" in history_entry:
                raw_usage = history_entry["usage"]

            if raw_usage:
                # Check for various common key/attribute names.
                usage.input_tokens = (
                    getattr(raw_usage, "prompt_tokens", None)
                    or getattr(raw_usage, "input_tokens", None)
                    or (raw_usage.get("prompt_tokens") if isinstance(raw_usage, dict) else None)
                    or (raw_usage.get("input_tokens") if isinstance(raw_usage, dict) else None)
                )
                usage.output_tokens = (
                    getattr(raw_usage, "completion_tokens", None)
                    or getattr(raw_usage, "output_tokens", None)
                    or (raw_usage.get("completion_tokens") if isinstance(raw_usage, dict) else None)
                    or (raw_usage.get("output_tokens") if isinstance(raw_usage, dict) else None)
                )

        return usage
