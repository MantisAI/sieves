"""
Imports 3rd-party libraries required for distillation. If library can't be found, placeholder engines is imported
instead.
This allows us to import everything downstream without having to worry about optional dependencies. If a user specifies
a non-installed distillation framework, we terminate with an error.
"""

# mypy: disable-error-code="no-redef"

import warnings

_MISSING_WARNING = (
    "Warning: dependency `{missing_dependency}` could not be imported." "this dependency has been installed."
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
    import fastfit
except ModuleNotFoundError:
    fastfit = None

    warnings.warn(_MISSING_WARNING.format(missing_dependency="fastfit"))


try:
    import model2vec
except ModuleNotFoundError:
    model2vec = None

    warnings.warn(_MISSING_WARNING.format(missing_dependency="model2vec"))


__all__ = ["fastfit", "model2vec", "sentence_transformers", "setfit"]
