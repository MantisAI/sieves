"""Tasks."""

from sieves.tasks import predictive, preprocessing
from sieves.tasks.core import Task
from sieves.tasks.distillation.types import DistillationFramework
from sieves.tasks.optimization import Optimizer
from sieves.tasks.predictive import (
    NER,
    Classification,
    InformationExtraction,
    PIIMasking,
    QuestionAnswering,
    SentimentAnalysis,
    Summarization,
    Translation,
)
from sieves.tasks.predictive.core import PredictiveTask
from sieves.tasks.preprocessing import Chunking, Ingestion

__all__ = [
    "Chunking",
    "Classification",
    "DistillationFramework",
    "InformationExtraction",
    "Ingestion",
    "NER",
    "Optimizer",
    "SentimentAnalysis",
    "Summarization",
    "Translation",
    "QuestionAnswering",
    "PIIMasking",
    "Task",
    "predictive",
    "PredictiveTask",
    "preprocessing",
]
