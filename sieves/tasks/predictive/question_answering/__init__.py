"""Question answering task."""

from sieves.tasks.predictive.question_answering.core import QuestionAnswering
from sieves.tasks.predictive.schemas.question_answering import FewshotExample, Result

__all__ = ["QuestionAnswering", "FewshotExample", "Result"]
