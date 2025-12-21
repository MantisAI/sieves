"""Common types."""

from typing import Any

import pydantic


class ModelSettings(pydantic.BaseModel):
    """Settings for model for structured generation.

    :param init_kwargs: kwargs passed on to initialization of structured generator. Not all models use this.
    :param inference_kwargs: kwargs passed on to inference with structured generator.
    :param config_kwargs: Used only if supplied model is a DSPy model object, ignored otherwise. Optional kwargs
        supplied to dspy.configure().
    :param strict: If True, exception is raised if prompt response can't be parsed correctly.
    :param inference_mode: Specifies the inference mode for the model wrapper. If not provided, the model wrapper will
        use its default mode. The available modes depend on the selected model wrapper (e.g., DSPy supports 'predict',
        'chain_of_thought', 'react'; Outlines supports 'text', 'choice', 'regex', 'json').
    """

    init_kwargs: dict[str, Any] | None = None
    inference_kwargs: dict[str, Any] | None = None
    config_kwargs: dict[str, Any] | None = None
    strict: bool = True
    inference_mode: Any | None = None


class TokenUsage(pydantic.BaseModel):
    """Token usage for a model call.

    :param input_tokens: Number of input tokens.
    :param output_tokens: Number of output tokens.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
