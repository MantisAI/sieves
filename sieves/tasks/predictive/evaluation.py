"""Core evaluation logic and base classes."""

from __future__ import annotations

import dataclasses
from typing import Any

from sieves.data import Doc


@dataclasses.dataclass
class TaskEvaluationReport[T]:
    """Report containing evaluation results."""

    metrics: dict[str, float]
    task_id: str
    failures: list[Doc] = dataclasses.field(default_factory=list)

    def summary(self) -> str:
        """Return a string summary of the report.

        :return: Summary string.
        """
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in self.metrics.items())
        return f"Task: {self.task_id} | Metrics: {metrics_str} | Failures: {len(self.failures)}"


@dataclasses.dataclass
class PipelineEvaluationReport:
    """Report containing evaluation results for a pipeline."""

    reports: dict[str, TaskEvaluationReport[Any]]

    def summary(self) -> str:
        """Return a string summary of the pipeline report.

        :return: Summary string.
        """
        summaries = [report.summary() for report in self.reports.values()]
        return "\n".join(summaries)

    def __getitem__(self, task_id: str) -> TaskEvaluationReport[Any]:
        """Get report for specified task.

        :param task_id: Task ID.
        :return: Evaluation report.
        """
        return self.reports[task_id]
