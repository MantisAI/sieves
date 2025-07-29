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
    from . import dspy_
    from .dspy_ import DSPy
except ModuleNotFoundError:
    from . import missing as dspy_

    DSPy = None  # type: ignore[assignment,misc]
    warnings.warn(_MISSING_WARNING.format(missing_dependency="dspy"))


try:
    from . import glix_
    from .glix_ import GliX
except ModuleNotFoundError:
    from . import missing as glix_

    GliX = None  # type: ignore[assignment,misc]
    warnings.warn(_MISSING_WARNING.format(missing_dependency="gliner"))


try:
    from . import huggingface_
    from .huggingface_ import HuggingFace
except ModuleNotFoundError:
    from . import missing as huggingface_

    HuggingFace = None  # type: ignore[assignment,misc]
    warnings.warn(_MISSING_WARNING.format(missing_dependency="transformers"))


try:
    from . import instructor_
    from .instructor_ import Instructor
except ModuleNotFoundError:
    from . import missing as instructor_

    Instructor = None  # type: ignore[assignment,misc]
    warnings.warn(_MISSING_WARNING.format(missing_dependency="instructor"))


try:
    from . import langchain_
    from .langchain_ import LangChain
except ModuleNotFoundError:
    from . import missing as langchain_

    LangChain = None  # type: ignore[assignment,misc]
    warnings.warn(_MISSING_WARNING.format(missing_dependency="langchain"))


try:
    from . import ollama_
    from .ollama_ import Ollama
except ModuleNotFoundError:
    from . import missing as ollama_

    Ollama = None  # type: ignore[assignment,misc]
    warnings.warn(_MISSING_WARNING.format(missing_dependency="ollama"))


try:
    from . import outlines_
    from .outlines_ import Outlines
except ModuleNotFoundError:
    from . import missing as outlines_

    Outlines = None  # type: ignore[assignment,misc]
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
