"""Aspect-based sentiment analysis."""

from sieves.tasks.predictive.schemas.sentiment_analysis import FewshotExample, Result
from sieves.tasks.predictive.sentiment_analysis.core import SentimentAnalysis

__all__ = ["SentimentAnalysis", "FewshotExample", "Result"]
