"""Relation extraction task."""

from sieves.tasks.predictive.relation_extraction.core import RelationExtraction
from sieves.tasks.predictive.schemas.relation_extraction import (
    FewshotExample,
    RelationEntity,
    RelationTriplet,
    Result,
)

__all__ = ["RelationExtraction", "FewshotExample", "RelationEntity", "RelationTriplet", "Result"]
