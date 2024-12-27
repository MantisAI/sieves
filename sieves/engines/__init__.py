import enum

import outlines_engine

from .core import Engine, Executable, ExecutableResult, PromptSignature, PromptTemplate


class EngineType(enum.Enum):
    outlines = 0
    # dspy = 1
    # jsonformer = 2


__all__ = [
    "Engine",
    "EngineType",
    "Executable",
    "ExecutableResult",
    "outlines_engine",
    "PromptSignature",
    "PromptTemplate",
]
