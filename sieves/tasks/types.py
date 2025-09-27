"""Common types."""

from sieves.engines.engine_import import (
    dspy_,
    glix_,
    huggingface_,
    instructor_,
    langchain_,
    ollama_,
    outlines_,
    vllm_,
)

Model = (
    dspy_.Model
    | glix_.Model
    | huggingface_.Model
    | instructor_.Model
    | langchain_.Model
    | ollama_.Model
    | outlines_.Model
    | vllm_.Model
)
