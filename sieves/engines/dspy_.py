import enum
from collections.abc import Iterable
from typing import Any, Optional, TypeAlias

import dsp
import dspy

from sieves.engines.core import Engine, Executable

PromptSignature: TypeAlias = type[dspy.Signature] | type[dspy.Module]
Model: TypeAlias = dsp.LM | dspy.BaseLM
Result: TypeAlias = dspy.Prediction


class InferenceMode(enum.Enum):
    """Available inference modes.
    See https://dspy.ai/#__tabbed_2_6 for more information and examples.
    """

    # Default inference mode.
    predict = dspy.Predict
    # CoT-style inference.
    chain_of_thought = dspy.TypedChainOfThought
    # Agentic, i.e. with tool use.
    react = dspy.ReAct
    # For multi-stage pipelines within a task. This is handled differently than the other supported modules: dspy.Module
    # serves as both the signature as well as the inference generator.
    module = dspy.Module


class DSPy(Engine[PromptSignature, Result, Model, InferenceMode]):
    """Engine for DSPy."""

    def __init__(self, model: Model):
        """
        :param model: Model to run. Note: DSPy only runs with APIs. If you want to run a model locally from v2.5
            onwards, serve it with OLlama - see here: # https://dspy.ai/learn/programming/language_models/?h=models#__tabbed_1_5.
            In a nutshell:
            > curl -fsSL https://ollama.ai/install.sh | sh
            > ollama run MODEL_ID
            > `model = dspy.LM(MODEL_ID, api_base='http://localhost:11434', api_key='')`
        """
        super().__init__(model)
        dspy.configure(lm=model)

    @property
    def inference_modes(self) -> type[InferenceMode]:
        return InferenceMode

    def build_executable(
        self,
        inference_mode: InferenceMode,
        prompt_template: Optional[str],  # noqa: UP007
        prompt_signature: PromptSignature,
    ) -> Executable[Result]:
        def execute(values: Iterable[dict[str, Any]]) -> Iterable[Result]:
            # Handled differently than the other supported modules: dspy.Module serves as both the signature as well as
            # the inference generator.
            if inference_mode == InferenceMode.module:
                assert isinstance(prompt_signature, dspy.Module), ValueError(
                    "In inference mode 'module' the provided prompt signature has to be of type dspy.Module."
                )
                generator = inference_mode.value()
            else:
                assert issubclass(prompt_signature, dspy.Signature)
                generator = inference_mode.value(signature=prompt_signature)

            # Note: prompt template isn't used here explicitly, as DSPy expects the complete prompt of the signature's
            # fields.
            return (generator(**doc_values) for doc_values in values)

        return execute
