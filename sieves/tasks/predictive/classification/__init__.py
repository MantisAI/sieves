"""Classification task."""

from sieves.tasks.predictive.classification.core import Classification
from sieves.tasks.predictive.schemas.classification import (
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
