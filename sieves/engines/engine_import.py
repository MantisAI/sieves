"""Import 3rd-party libraries required for engines.

If library can't be found, placeholder engines is imported instead.

This allows us to import everything downstream without having to worry about optional dependencies. If a user specifies
an engine/model from a non-installed library, we terminate with an error.
"""

from sieves.engines import dspy_, gliner_, huggingface_, langchain_, outlines_
from sieves.engines.dspy_ import DSPy
from sieves.engines.gliner_ import GliNER
from sieves.engines.huggingface_ import HuggingFace
from sieves.engines.langchain_ import LangChain
from sieves.engines.outlines_ import Outlines

__all__ = [
    "dspy_",
    "DSPy",
    "gliner_",
    "GliNER",
    "huggingface_",
    "HuggingFace",
    "langchain_",
    "LangChain",
    "outlines_",
    "Outlines",
]
