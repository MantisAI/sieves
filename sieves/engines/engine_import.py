"""
Imports 3rd-party libraries required for engines. If library can't be found, placeholder engines is imported instead.
This allows us to import everything downstream without having to worry about optional dependencies. If a user specifies
an engine/model from a non-installed library, we terminate with an error.
"""

# mypy: disable-error-code="no-redef"

import warnings

_MISSING_WARNING = (
    "Warning: engine dependency `{missing_dependency}` could not be imported. The corresponding engines won't work "
    "unless this dependency has been installed."
)


try:
    from dspy_ import DSPy

    from . import dspy_
except ModuleNotFoundError:
    from . import missing as dspy_

    DSPy = None
    warnings.warn(_MISSING_WARNING.format(missing_dependency="dspy"))


try:
    from glix_ import GliX

    from . import glix_
except ModuleNotFoundError:
    from . import missing as glix_

    GliX = None
    warnings.warn(_MISSING_WARNING.format(missing_dependency="gliner"))


try:
    from huggingface_ import HuggingFace

    from . import huggingface_
except ModuleNotFoundError:
    from . import missing as huggingface_

    HuggingFace = None
    warnings.warn(_MISSING_WARNING.format(missing_dependency="transformers"))


try:
    from instructor_ import Instructor

    from . import instructor_
except ModuleNotFoundError:
    from . import missing as instructor_

    Instructor = None
    warnings.warn(_MISSING_WARNING.format(missing_dependency="instructor"))


try:
    from langchain_ import LangChain

    from . import langchain_
except ModuleNotFoundError:
    from . import missing as langchain_

    LangChain = None
    warnings.warn(_MISSING_WARNING.format(missing_dependency="langchain"))


try:
    from ollama_ import Ollama

    from . import ollama_
except ModuleNotFoundError:
    from . import missing as ollama_

    Ollama = None
    warnings.warn(_MISSING_WARNING.format(missing_dependency="ollama"))


try:
    from outlines_ import Outlines

    from . import outlines_
except ModuleNotFoundError:
    from . import missing as outlines_

    Outlines = None
    warnings.warn(_MISSING_WARNING.format(missing_dependency="outlines"))


try:
    from vllm_ import VLLM

    from . import vllm_
except ModuleNotFoundError:
    from . import missing as vllm_

    VLLM = None
    warnings.warn(_MISSING_WARNING.format(missing_dependency="vllm"))


__all__ = [
    "dspy_",
    "DSPy",
    "glix_",
    "GliX",
    "huggingface_",
    "HuggingFace",
    "instructor_",
    "Instructor",
    "langchain_",
    "LangChain",
    "ollama_",
    "Ollama",
    "outlines_",
    "Outlines",
    "vllm_",
    "VLLM",
]
