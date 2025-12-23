"""Information extraction task."""

from sieves.tasks.predictive.information_extraction.core import InformationExtraction
from sieves.tasks.predictive.schemas.information_extraction import (
    FewshotExampleMulti,
    FewshotExampleSingle,
    ResultMulti,
    ResultSingle,
)

__all__ = ["InformationExtraction", "FewshotExampleMulti", "FewshotExampleSingle", "ResultMulti", "ResultSingle"]
