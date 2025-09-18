"""
Imports 3rd-party libraries required for distillation. If library can't be found, placeholder engines is imported
instead.
This allows us to import everything downstream without having to worry about optional dependencies. If a user specifies
a non-installed distillation framework, we terminate with an error.
"""

# mypy: disable-error-code="no-redef"

import warnings

_MISSING_WARNING = (
    "Warning: engine dependency `{missing_dependency}` could not be imported. The corresponding engines won't work "
    "unless this dependency has been installed."
)


try:
    import sentence_transformers
except ModuleNotFoundError:
    sentence_transformers = None

    warnings.warn(_MISSING_WARNING.format(missing_dependency="sentence_transformers"))


try:
    import setfit
except ModuleNotFoundError:
    setfit = None

    warnings.warn(_MISSING_WARNING.format(missing_dependency="setfit"))

try:
    import model2vec
    import model2vec.train
except ModuleNotFoundError:
    model2vec = None

    warnings.warn(_MISSING_WARNING.format(missing_dependency="model2vec"))


__all__ = ["model2vec", "sentence_transformers", "setfit"]
