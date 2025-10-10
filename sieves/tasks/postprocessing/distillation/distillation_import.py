"""Import 3rd-party libraries required for distillation.

If library can't be found, placeholder engines is imported instead.

This allows us to import everything downstream without having to worry about optional dependencies. If a user specifies
a non-installed distillation framework, we terminate with an error.
"""

# mypy: disable-error-code="no-redef"

import warnings

_missing_dependencies: list[str] = []


try:
    import sentence_transformers
except ModuleNotFoundError:
    sentence_transformers = None

    _missing_dependencies.append("sentence_transformers")

try:
    import setfit
except ModuleNotFoundError:
    setfit = None

    _missing_dependencies.append("setfit")

try:
    import model2vec
    import model2vec.train
except ModuleNotFoundError:
    model2vec = None

    _missing_dependencies.append("model2vec")

if len(_missing_dependencies):
    warnings.warn(
        "Warning: distillation dependency [{deps}] could not be imported. Distilling with these tools requires them to "
        "be installed.".format(deps=", ".join(_missing_dependencies))
    )

__all__ = ["model2vec", "sentence_transformers", "setfit"]
