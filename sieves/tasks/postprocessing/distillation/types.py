from __future__ import annotations

import enum
from typing import Literal


class DistillationFramework(enum.Enum):
    model2vec = "model2vec"
    sentence_transformers = "sentence_transformers"
    setfit = "setfit"

    @classmethod
    def all(cls) -> tuple[DistillationFramework, ...]:
        """Returns all available engine types.
        :return tuple[EngineType, ...]: All available engine types.
        """
        return tuple(dist_type for dist_type in DistillationFramework)


DistillationFrameworkLiteral = Literal[*DistillationFramework.all()]  # type: ignore[valid-type]
