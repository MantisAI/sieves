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
except ModuleNotFoundError:
    from . import missing as dspy_

    warnings.warn(_MISSING_WARNING.format(missing_dependency="dspy"))


try:
    from . import glix_
except ModuleNotFoundError:
    from . import missing as glix_

    warnings.warn(_MISSING_WARNING.format(missing_dependency="gliner"))


try:
    from . import huggingface_
except ModuleNotFoundError:
    from . import missing as huggingface_

    warnings.warn(_MISSING_WARNING.format(missing_dependency="transformers"))


try:
    from . import instructor_
except ModuleNotFoundError:
    from . import missing as instructor_

    warnings.warn(_MISSING_WARNING.format(missing_dependency="instructor"))


try:
    from . import langchain_
except ModuleNotFoundError:
    from . import missing as langchain_

    warnings.warn(_MISSING_WARNING.format(missing_dependency="langchain"))


try:
    from . import ollama_
except ModuleNotFoundError:
    from . import missing as ollama_

    warnings.warn(_MISSING_WARNING.format(missing_dependency="ollama"))


try:
    from . import outlines_
except ModuleNotFoundError:
    from . import missing as outlines_

    warnings.warn(_MISSING_WARNING.format(missing_dependency="outlines"))


try:
    from . import vllm_
except ModuleNotFoundError:
    from . import missing as vllm_

    warnings.warn(_MISSING_WARNING.format(missing_dependency="vllm"))


__all__ = ["dspy_", "glix_", "huggingface_", "instructor_", "langchain_", "ollama_", "outlines_", "vllm_"]
