"""Information extraction task."""

from .core import FewshotExample, Translation, _TaskPromptSignature, _TaskResult

__all__ = ["Translation", "FewshotExample", "_TaskResult", "_TaskPromptSignature"]
