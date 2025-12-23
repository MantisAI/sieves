"""Classification task."""

from sieves.tasks.predictive.classification.core import Classification
from sieves.tasks.predictive.classification.schemas import (
    FewshotExampleMultiLabel,
    FewshotExampleSingleLabel,
    ResultMultiLabel,
    ResultSingleLabel,
)

__all__ = [
    "Classification",
    "FewshotExampleMultiLabel",
    "FewshotExampleSingleLabel",
    "ResultMultiLabel",
    "ResultSingleLabel",
]
