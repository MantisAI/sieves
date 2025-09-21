"""Tasks."""

from . import predictive, preprocessing
from .core import Task
from .postprocessing import Distillation, DistillationFramework
from .predictive import (
    Classification,
    InformationExtraction,
    PIIMasking,
    QuestionAnswering,
    SentimentAnalysis,
    Summarization,
    Translation,
)
from .predictive.core import PredictiveTask
from .preprocessing import Chunking, Ingestion

__all__ = [
    "Chunking",
    "Classification",
    "Distillation",
    "DistillationFramework",
    "InformationExtraction",
    "Ingestion",
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
