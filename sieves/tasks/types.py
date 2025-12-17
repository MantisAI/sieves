"""Common types."""

from sieves.model_wrappers import (
    dspy_,
    gliner_,
    huggingface_,
    langchain_,
    outlines_,
)

Model = dspy_.Model | gliner_.Model | huggingface_.Model | langchain_.Model | outlines_.Model
