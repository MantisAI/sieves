"""Import 3rd-party libraries required for engines.

If library can't be found, placeholder engines is imported instead.

This allows us to import everything downstream without having to worry about optional dependencies. If a user specifies
an engine/model from a non-installed library, we terminate with an error.
"""

# mypy: disable-error-code="no-redef"

import warnings

from .missing import MissingEngine

_missing_dependencies: list[str] = []


try:
    from . import dspy_
    from .dspy_ import DSPy
except ModuleNotFoundError:
    from . import missing as dspy_

    DSPy = MissingEngine  # type: ignore[misc,assignment]
    _missing_dependencies.append("dspy")


try:
    from . import glix_
    from .glix_ import GliX
except ModuleNotFoundError:
    from . import missing as glix_

    GliX = MissingEngine  # type: ignore[misc,assignment]
    _missing_dependencies.append("gliner")


try:
    from . import huggingface_
    from .huggingface_ import HuggingFace
except ModuleNotFoundError:
    from . import missing as huggingface_

    HuggingFace = MissingEngine  # type: ignore[misc,assignment]
    _missing_dependencies.append("transformers")


try:
    from . import langchain_
    from .langchain_ import LangChain
except ModuleNotFoundError:
    from . import missing as langchain_

    LangChain = MissingEngine  # type: ignore[misc,assignment]
    _missing_dependencies.append("langchain")


try:
    from . import outlines_
    from .outlines_ import Outlines
except ModuleNotFoundError:
    from . import missing as outlines_

    Outlines = MissingEngine  # type: ignore[misc,assignment]
    _missing_dependencies.append("outlines")


if len(_missing_dependencies):
    warnings.warn(
        "Warning: structured generation dependencies [{deps}] could not be imported. Generating with them requires them"
        " to be installed.".format(deps=", ".join(_missing_dependencies))
    )


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
