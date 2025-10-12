"""Import 3rd-party libraries required for distillation.

If library can't be found, placeholder engines is imported instead.

This allows us to import everything downstream without having to worry about optional dependencies. If a user specifies
a non-installed distillation framework, we terminate with an error.
"""

# mypy: disable-error-code="no-redef"

try:
    import sentence_transformers
except ModuleNotFoundError:
    sentence_transformers = None


try:
    import setfit
except ModuleNotFoundError:
    setfit = None


try:
    import model2vec
    import model2vec.train
except ModuleNotFoundError:
    model2vec = None


__all__ = ["model2vec", "sentence_transformers", "setfit"]
