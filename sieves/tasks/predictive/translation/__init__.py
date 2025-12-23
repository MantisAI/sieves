"""Translation task."""

from sieves.tasks.predictive.schemas.translation import (
    FewshotExample,
    Result,
    TaskPromptSignature,
    TaskResult,
)
from sieves.tasks.predictive.translation.core import Translation

__all__ = ["Translation", "FewshotExample", "Result", "TaskResult", "TaskPromptSignature"]
