"""Summarization task."""

from sieves.tasks.predictive.summarization.core import Summarization
from sieves.tasks.predictive.summarization.schemas import FewshotExample, Result

__all__ = ["Summarization", "FewshotExample", "Result"]
