"""Import 3rd-party libraries required for distillation.

If library can't be found, placeholder model wrappers is imported instead.

This allows us to import everything downstream without having to worry about optional dependencies. If a user specifies
a non-installed distillation framework, we terminate with an error.
"""

# mypy: disable-error-code="no-redef"

try:
    import sentence_transformers
except (ModuleNotFoundError, ImportError):
    sentence_transformers = None


try:
    import setfit
except (ModuleNotFoundError, ImportError):
    setfit = None


try:
    import model2vec
    import model2vec.train
except (ModuleNotFoundError, ImportError):
    model2vec = None


__all__ = ["model2vec", "sentence_transformers", "setfit"]
