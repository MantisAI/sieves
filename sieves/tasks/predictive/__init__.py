"""Predictive tasks."""

from sieves.tasks.predictive.classification import Classification
from sieves.tasks.predictive.core import PredictiveTask
from sieves.tasks.predictive.information_extraction import InformationExtraction
from sieves.tasks.predictive.ner import NER
from sieves.tasks.predictive.pii_masking import PIIMasking
from sieves.tasks.predictive.question_answering import QuestionAnswering
from sieves.tasks.predictive.relation_extraction import RelationExtraction
from sieves.tasks.predictive.sentiment_analysis import SentimentAnalysis
from sieves.tasks.predictive.summarization import Summarization
from sieves.tasks.predictive.translation import Translation

__all__ = [
    "Classification",
    "InformationExtraction",
    "SentimentAnalysis",
    "Summarization",
    "Translation",
    "NER",
    "PIIMasking",
    "PredictiveTask",
    "QuestionAnswering",
    "RelationExtraction",
]
