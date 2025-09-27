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
    :param batch_size: Batch size in processing prompts. -1 will batch all documents in one go. Not all engines
        support batching.
    """

    init_kwargs: dict[str, Any] | None = None
    inference_kwargs: dict[str, Any] | None = None
    config_kwargs: dict[str, Any] | None = None
    strict_mode: bool = False
    batch_size: int = -1
