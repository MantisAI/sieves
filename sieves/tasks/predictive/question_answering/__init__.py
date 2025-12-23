"""Question answering task."""

from sieves.tasks.predictive.question_answering.core import QuestionAnswering
from sieves.tasks.predictive.question_answering.schemas import FewshotExample, Result

__all__ = ["QuestionAnswering", "FewshotExample", "Result"]
