"""Summarization task."""

from sieves.tasks.predictive.schemas.summarization import FewshotExample, Result
from sieves.tasks.predictive.summarization.core import Summarization

__all__ = ["Summarization", "FewshotExample", "Result"]
