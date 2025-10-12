"""Import 3rd-party libraries required for engines.

If library can't be found, placeholder engines is imported instead.

This allows us to import everything downstream without having to worry about optional dependencies. If a user specifies
an engine/model from a non-installed library, we terminate with an error.
"""

from .missing import MissingEngine

try:
    from . import dspy_
    from .dspy_ import DSPy
except ModuleNotFoundError:
    from . import missing as dspy_

    DSPy = MissingEngine  # type: ignore[misc,assignment]


try:
    from . import glix_
    from .glix_ import GliX
except ModuleNotFoundError:
    from . import missing as glix_

    GliX = MissingEngine  # type: ignore[misc,assignment]


try:
    from . import huggingface_
    from .huggingface_ import HuggingFace
except ModuleNotFoundError:
    from . import missing as huggingface_

    HuggingFace = MissingEngine  # type: ignore[misc,assignment]


try:
    from . import langchain_
    from .langchain_ import LangChain
except ModuleNotFoundError:
    from . import missing as langchain_

    LangChain = MissingEngine  # type: ignore[misc,assignment]


try:
    from . import outlines_
    from .outlines_ import Outlines
except ModuleNotFoundError:
    from . import missing as outlines_

    Outlines = MissingEngine  # type: ignore[misc,assignment]


__all__ = [
    "dspy_",
    "DSPy",
    "glix_",
    "GliX",
    "huggingface_",
    "HuggingFace",
    "langchain_",
    "LangChain",
    "outlines_",
    "Outlines",
]
