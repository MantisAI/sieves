"""Common types."""

from typing import Any

import pydantic


class GenerationSettings(pydantic.BaseModel):
    """Settings for structured generation.

    :param init_kwargs: kwargs passed on to initialization of structured generator. Not all engines use this - ignored
        otherwise.
    :param inference_kwargs: kwargs passed on to inference with structured generator.
    :param config_kwargs: Used only if supplied model is a DSPy model object, ignored otherwise. Optional kwargs
        supplied to dspy.configure().
    :param strict_mode: If True, exception is raised if prompt response can't be parsed correctly.
    :param inference_mode: Specifies the inference mode for the engine. If not provided, the engine will use its
        default mode. The available modes depend on the selected engine (e.g., DSPy supports 'predict',
        'chain_of_thought', 'react'; Outlines supports 'text', 'choice', 'regex', 'json').
    """

    init_kwargs: dict[str, Any] | None = None
    inference_kwargs: dict[str, Any] | None = None
    config_kwargs: dict[str, Any] | None = None
    strict_mode: bool = False
    inference_mode: Any | None = None
