"""Common types."""

from sieves.engines.engine_import import (
    dspy_,
    glix_,
    huggingface_,
    langchain_,
    outlines_,
)

Model = dspy_.Model | glix_.Model | huggingface_.Model | langchain_.Model | outlines_.Model
