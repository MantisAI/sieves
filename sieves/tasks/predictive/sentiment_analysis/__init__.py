"""Aspect-based sentiment analysis."""

from sieves.tasks.predictive.sentiment_analysis.core import SentimentAnalysis
from sieves.tasks.predictive.sentiment_analysis.schemas import FewshotExample, Result

__all__ = ["SentimentAnalysis", "FewshotExample", "Result"]
